"""
bugs found:
    - the multi band exp does not propagate wcs, filter labels
    - clone() does not copy over the psfs

feature requests for DM
    - have MultibandExposure keep track of wcs, filter labels, etc.
    - footprint addTo and subtractFrom methods so we don't need
      twice the memory
    - clone() copy psfs
"""
from contextlib import contextmanager
import ngmix
import numpy as np
import lsst.afw.detection as afw_det
import lsst.afw.table as afw_table
import lsst.afw.image as afw_image
from lsst.afw.image import MultibandExposure
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
)
from lsst.meas.extensions.scarlet import (
    ScarletDeblendTask, ScarletDeblendConfig,
)
import lsst.geom as geom

from lsst.pex.exceptions import InvalidParameterError, LengthError
from . import vis
from . import util
from .defaults import DEFAULT_THRESH
from . import procflags
from .lsst_measure import (
    get_output_struct, get_meas_type, get_ormask, measure_one,
    MissingDataError,
)
import logging

LOG = logging.getLogger('lsst_measure_scarlet')


def detect_and_deblend(
    mbexp,
    thresh=DEFAULT_THRESH,
):
    """
    run detection and the scarlet deblender

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    thresh: float
        The detection threshold in units of the sky noise

    Returns
    -------
    sources, detexp
        Sources is an lsst.afw.table.SourceCatalog
        detexp is lsst.afw.image.ExposureF
    """

    schema = afw_table.SourceTable.makeMinimalSchema()

    # note we won't run any of these measurements, but it is needed so that
    # getCentroid will return the peak position rather than NaN.
    # I think it modifies the schema and sets defaults

    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_PsfFlux",
        "base_SkyCoord",
    ]

    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None

    # goes with SdssShape above
    meas_config.slots.shape = None

    # fix odd issue where it thinks things are near the edge
    meas_config.plugins['base_SdssCentroid'].binmax = 1

    _ = SingleFrameMeasurementTask(
        config=meas_config,
        schema=schema,
    )

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = thresh
    detection_task = SourceDetectionTask(config=detection_config)

    # configure the scarlet deblender
    #
    # this must occur directly before any tasks are run because schema is
    # modified in place by tasks, and the constructor does a check that
    # fails if we construct it separately

    # leave minSNR at 50; that just controls how many components can be
    # modeled, and in any case it drops back to 1 extended component
    # which is OK
    # default is sourceModel is 'double'

    deblend_config = ScarletDeblendConfig()
    deblend_task = ScarletDeblendTask(
        config=deblend_config,
        schema=schema,
    )

    table = afw_table.SourceTable.make(schema)

    if len(mbexp.singles) > 1:
        detexp = util.coadd_exposures(mbexp.singles)
    else:
        detexp = mbexp.singles[0]

    result = detection_task.run(table, detexp)

    if result is not None:
        sources = deblend_task.run(mbexp, result.sources)
    else:
        # sources = []
        sources = None

    return sources, detexp


def measure(
    mbexp,
    detexp,
    sources,
    fitter,
    rng,
    stamp_size,
    show=False,
):

    """
    run measurements on the input exposures

    We send both mbexp and the original exposures because the MultibandExposure
    does not keep track of the wcs

    Create mbexp with util.get_mbexp

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    detexp: Exposure
        The detection exposure, used for getting ormasks. This is returned
        by lsst_measure_scarlet.detect_and_deblend
    sources: list of sources
        From a detection task
    fitter: e.g. ngmix.gaussmom.GaussMom or ngmix.ksigmamom.KSigmaMom
        For calculating moments
    rng: np.random.RandomState
        Random number generator for the centroid algorithm
    stamp_size: int
        Size for postage stamps
    show: bool, optional
        If set to True, show images

    Returns
    -------
    array of results, with flags, positions, shapes, etc. or None
    if there were no objects to measure
    """

    wcs = mbexp.singles[0].getWcs()

    subtractor = ModelSubtractor(
        mbexp=mbexp,
        sources=sources,
    )

    if show:
        model = subtractor.get_full_model()
        vis.compare_mbexp(mbexp=mbexp, model=model)

    results = []

    # assumption: sources same for all bands
    band_sources = sources[subtractor.filters[0]]

    # this gets everything that is a parent
    parents = band_sources.getChildren(0)

    for meas_parent_record in parents:
        LOG.debug('-'*70)
        LOG.debug('parent id: %d', meas_parent_record.getId())

        children = band_sources.getChildren(meas_parent_record.getId())
        nchild = len(children)
        LOG.debug(
            'processing %d %s',
            nchild, 'children' if nchild > 1 else 'child'
        )
        results += _process_sources(
            subtractor=subtractor, sources=children, stamp_size=stamp_size,
            detexp=detexp, wcs=wcs, fitter=fitter, rng=rng, show=show,
        )

    if len(results) > 0:
        results = np.hstack(results)
    else:
        results = None

    return results


def _process_sources(
    subtractor, sources, stamp_size, wcs, fitter, detexp, rng, show=False,
):
    results = []
    for source in sources:
        res = _process_source(
            subtractor=subtractor, source=source, stamp_size=stamp_size,
            detexp=detexp, wcs=wcs, fitter=fitter, rng=rng, show=show,
        )
        results.append(res)

    return results


def _process_source(
    subtractor, source, stamp_size, wcs, fitter, detexp, rng, show=False,
):
    source_id = source.getId()

    ormask = get_ormask(source=source, exposure=detexp)
    exp_bbox = detexp.getBBox()

    with subtractor.add_source(source_id):

        if show:
            vis.show_mbexp(
                subtractor.mbexp, mess=f'source {source_id} added'
            )

        try:
            stamp_mbexp = subtractor.get_stamp(
                source_id, stamp_size=stamp_size,
            )
            # make sure wcs is getting propagated
            assert wcs == stamp_mbexp.singles[0].wcs

            if show:
                ostamp_mbexp = subtractor.get_stamp(
                    source_id, stamp_size=stamp_size, type='original',
                )
                model_mbexp = subtractor.get_stamp(
                    source_id, stamp_size=stamp_size, type='model',
                )
                vis.show_three_mbexp(
                    [ostamp_mbexp, stamp_mbexp, model_mbexp],
                    labels=['original', 'deblended', 'model']
                )

            # TODO make codes work with MultibandExposure rather than on a
            # coadd of the bands

            coadded_stamp_exp = util.coadd_exposures(stamp_mbexp.singles)
            obs = extract_obs(subim=coadded_stamp_exp, source=source)
            if obs is None:
                LOG.info('skipping object with all zero weights')
                ores = {'flags': procflags.ZERO_WEIGHTS}
                pres = {'flags': procflags.NO_ATTEMPT}
            else:
                try:

                    # this updates the jacobian center as well as the
                    # meta['orig_cen'] pixel location in the original image

                    find_and_set_center(obs=obs, rng=rng)

                    pres = measure_one(obs=obs.psf, fitter=fitter)
                    ores = measure_one(obs=obs, fitter=fitter)

                except CentroidFail as err:
                    LOG.info(str(err))
                    ores = {'flags': procflags.CENTROID_FAIL}
                    pres = {'flags': procflags.NO_ATTEMPT}

        except LengthError as e:
            # bounding box did not fit. TODO keep output with flag set
            LOG.info(e)

            # note the context manager properly handles a return
            ores = {'flags': procflags.BBOX_HITS_EDGE}
            pres = {'flags': procflags.NO_ATTEMPT}
            obs = None

        res = get_output(
            obs=obs, wcs=wcs, fitter=fitter, source=source, res=ores, pres=pres,
            ormask=ormask, stamp_size=stamp_size, exp_bbox=exp_bbox,
        )

    return res


class ModelSubtractor(object):
    """
    Create an image with all models subtracted, which is stored in the .mbexp
    attribute

    Provides a method to add back a source.  This works in a context manager to
    maintain data consistency.

    Provides methods to get a postage stamp for a deblended source.  One can
    also get a stamp for the model or the original image.

    You an also generate the full model of the image.

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        A representation of the multi-band data set.
        Create one of these with
            descwl_model_subtractor.get_mbexp(exposure_list)
    sources: dict of lsst.afw.table.SourceCatalog
        This is the output of the ScarletDeblendTask, a dict of
        lsst.afw.table.SourceCatalog keyed by band

    TODO
    ----
        - move this to its own repo
        - This code would use half the memory if heavy footprints supported
        operations like .addTo(image) or .subtractFrom(image) which should be
        essentially the same as the code in .insert

    Examples
    ---------

    subtractor = ModelSubtractor(exposure, sources)

    # add back one model; since the image had data-model this restores the
    # pixels for the object of interest, but with models of other objects
    # subtracted

    with subtractor.add_source(source_id):
        # full MultibandExposure is in subtractor.mbexp

        stamp = subtractor.get_stamp(source_id, stamp_size=48)

    # model of the entire data set as a MultibandExposure
    full_model = subtractor.get_full_model()

    # model of one source
    model = subtractor.get_model(source_id, stamp_size=48)
    """
    def __init__(self, mbexp, sources):
        assert isinstance(mbexp, MultibandExposure)
        assert isinstance(sources, dict)

        self.orig = mbexp
        self.filters = mbexp.filters

        self.sources = sources
        self.source_ids = set()
        for source in sources[list(sources.keys())[0]]:
            self.source_ids.add(source.getId())

        self.mbexp = util.copy_mbexp(mbexp)
        for band in self.filters:
            psf = util.try_clone_psf(self.orig[band].getPsf())
            self.mbexp[band].setPsf(psf)

        # we need a scratch array because heavy footprings don't
        # have addTo or subtractFrom methods
        self.scratch = util.copy_mbexp(mbexp, clear=True)

        self._set_footprints()
        self._build_heavies()
        self._build_subtracted_image()

    @contextmanager
    def add_source(self, source_id):
        """
        Open a with context that yields the image with all objects
        subtracted except the specified one.

        since the image had data-model this restores the pixels for the object
        of interest, minus models of other objects

        with subtractor.add_source(source_id):
            # do something with subtractor.mbexp

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()

        Yields
        -------
        ExposureF, although more typically one uses the .mbexp attribute
        """
        if source_id not in self.source_ids:
            raise ValueError(f'source {source_id} is not in the source list')

        self._add_or_subtract_source(source_id, 'add')
        # self._add_or_subtract_source_new(source_id, 'add')
        try:
            yield self.mbexp
        finally:
            self._add_or_subtract_source(source_id, 'subtract')
            # self._add_or_subtract_source_new(source_id, 'subtract')

    def _add_or_subtract_source(self, source_id, type):
        mbexp = self.mbexp
        scratch = self.scratch

        bbox = self.get_bbox(source_id)

        for band in self.filters:
            # Because footprints can only be used to *replace* pixels, we do so
            # on a scratch image and then subtract that from the model image

            heavy_fp = self.heavies[band][source_id]
            heavy_fp.insert(scratch[band].image)

            if type == 'add':
                mbexp[band].image[bbox] += scratch[band].image[bbox]
            else:
                mbexp[band].image[bbox] -= scratch[band].image[bbox]

            scratch[band].image[bbox] = 0

    def _add_or_subtract_source_new(self, source_id, type):
        mbexp = self.mbexp
        scratch = self.scratch

        bbox = self.get_bbox(source_id)

        for band in self.filters:
            # Because footprints can only be used to *replace* pixels, we do so
            # on a scratch image and then subtract that from the model image

            heavy_fp = self.heavies[band][source_id]
            heavy_fp.insert(scratch[band].image)

            if type == 'add':
                heavy_fp.addTo(mbexp[band].image[bbox])
            else:
                heavy_fp.subtractFrom(mbexp[band].image[bbox])

            scratch[band].image[bbox] = 0

    def get_stamp(
        self, source_id, stamp_size=None, clip=False, type='deblended',
    ):
        """
        Get a postage stamp exposure at the location of the specified source.
        The pixel data are copied.

        If you want the object to be in the image, use this method within
        an add_source context

        with subtractor.add_source(source_id):
            stamp = subtractor.get_stamp(source_id)

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()
        stamp_size: int
            If sent, a bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

            Only relevant if stamp_size is sent.  Default False
        type: str, optional
            'deblended', 'original', 'model'.  Default is 'deblended'.

            'deblended' means whatever is in the current subtracted images.
                If the user is in the add_source() context it will contain
                the source data because the model will have been added back in

            'original' means a stamp from the original data

            'model' means the model for object
                You can also use get_model() to get the model

        Returns
        -------
        ExposureF
        """

        assert type in ['deblended', 'original', 'model'], (
            'type must be one of deblended, original or model'
        )

        if type == 'model':
            return self.get_model(source_id, stamp_size=stamp_size, clip=clip)

        if source_id not in self.source_ids:
            raise ValueError(f'source {source_id} is not in the source list')

        bbox = self.get_bbox(source_id, stamp_size=stamp_size, clip=clip)

        if type == 'original':
            mbexp = self.orig
        else:
            mbexp = self.mbexp

        exposures = [mbexp[band][bbox] for band in self.filters]
        # return MultibandExposure.fromExposures(self.filters, exposures)
        return util.get_mbexp(exposures)

    def get_model(self, source_id, stamp_size=None, clip=False):
        """
        Get a postage stamp exposure at the location of the specified source,
        containing the model rather than data.

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()
        stamp_size: int
            If sent, a bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

            Only relevant if stamp_size is sent.  Default False

        Returns
        -------
        ExposureF
        """

        if source_id not in self.source_ids:
            raise ValueError(f'source {source_id} is not in the source list')

        scratch = self.scratch
        heavies = self.heavies

        bbox = self.get_bbox(source_id, stamp_size=stamp_size, clip=clip)

        exposures = []
        for band in self.filters:

            heavy_fp = heavies[band][source_id]
            heavy_fp.insert(scratch[band].image)

            model_exp = afw_image.ExposureF(scratch[band][bbox], deep=True)

            scratch[band].image[bbox] = 0

            exposures.append(model_exp)

        # return MultibandExposure.fromExposures(self.filters, exposures)
        return util.get_mbexp(exposures)

    def get_full_model(self):
        """
        Get a full model image of all sources.

        Returns
        -------
        ExposureF
        """
        heavies = self.heavies
        scratch = self.scratch

        model = util.copy_mbexp(self.mbexp, clear=True)

        for band, sources in self.sources.items():
            LOG.debug('-'*70)
            LOG.debug(f'band: {band}')

            parents = sources.getChildren(0)
            for parent_record in parents:
                LOG.debug('parent id: %d', parent_record.getId())

                children = sources.getChildren(parent_record.getId())
                nchild = len(children)
                LOG.debug(
                    'processing %d %s',
                    nchild, 'children' if nchild > 1 else 'child',
                )

                for child in children:
                    child_id = child.getId()
                    heavy_fp = heavies[band][child_id]
                    heavy_fp.insert(scratch[band].image)

                    bbox = self.get_bbox(child_id)
                    model[band].image[bbox] += scratch[band].image[bbox]
                    scratch[band].image[bbox] = 0

        return model

    def get_bbox(self, source_id, stamp_size=None, clip=False):
        """
        Get a bounding box at the location of the specified source.

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()
        stamp_size: int
            If sent, a bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

            Only relevant if stamp_size is sent.  Default False

        Returns
        -------
        lsst.geom.Box2I
        """

        if source_id not in self.source_ids:
            raise ValueError(f'source {source_id} is not in the source list')

        # assumption: bounding boxes same in all bands
        band = self.filters[0]

        if stamp_size is not None:
            parent_id, fp = self.footprints[band][source_id]
            peak = fp.getPeaks()[0]

            # note we previously had -0.5 on each of these based on Bob's code
            x_peak, y_peak = peak.getIx(), peak.getIy()

            bbox = geom.Box2I(
                geom.Point2I(x_peak, y_peak),
                geom.Extent2I(1, 1),
            )
            bbox.grow(stamp_size // 2)

            exp_bbox = self.mbexp.getBBox()
            if clip:
                bbox.clip(exp_bbox)
            else:
                if not exp_bbox.contains(bbox):
                    raise LengthError(
                        f'requested stamp size {stamp_size} for source '
                        f'{source_id} does not fit into the exposoure.  '
                        f'Use clip=True to clip the bbox to fit'
                    )

        else:
            parent_id, fp = self.footprints[band][source_id]
            bbox = fp.getBBox()

        return bbox

    def _set_footprints(self):
        self.footprints = {}
        for band, sources in self.sources.items():
            self.footprints[band] = {
                source.getId(): (source.getParent(), source.getFootprint())
                for source in sources
            }

    def _build_heavies(self):
        self.heavies = {}
        for band, footprints in self.footprints.items():

            self.heavies[band] = {}

            for id, fp in footprints.items():
                if fp[1].isHeavy():
                    self.heavies[band][id] = fp[1]
                elif fp[0] == 0:
                    self.heavies[band][id] = afw_det.makeHeavyFootprint(
                        fp[1], self.mbexp[band].maskedImage,
                    )

    def _build_subtracted_image(self):
        heavies = self.heavies
        mbexp = self.mbexp
        scratch = self.scratch

        for band, sources in self.sources.items():
            LOG.debug('-'*70)
            LOG.debug(f'band: {band}')

            parents = sources.getChildren(0)
            for parent_record in parents:
                LOG.debug('parent id: %d', parent_record.getId())

                children = sources.getChildren(parent_record.getId())
                nchild = len(children)
                LOG.debug(
                    'processing %d %s',
                    nchild, 'children' if nchild > 1 else 'child',
                )

                for child in children:
                    child_id = child.getId()
                    heavy_fp = heavies[band][child_id]
                    heavy_fp.insert(scratch[band].image)

                    bbox = self.get_bbox(source_id=child_id)
                    # mbexp[band].image[bbox] -= scratch[band].image[bbox]
                    mbexp[band].image[bbox] -= scratch[band].image[bbox]
                    scratch[band].image[bbox] = 0


def extract_obs(subim, source):
    """
    convert an exposure object into an ngmix.Observation, including
    a psf observation.

    the center for the jacobian will be at the peak location, an integer
    location.

    Parameters
    ----------
    imobj: lsst.afw.image.Exposure
        An Exposure object, e.g. ExposureF
    source: lsst.afw.table.SourceRecord
        A source record from detection/deblending

    returns
    --------
    obs: ngmix.Observation
        The Observation unless all the weight are zero, in which
        case None is returned
    """

    im = subim.image.array

    wt = _extract_weight(subim)
    if np.all(wt <= 0):
        return None

    maskobj = subim.mask
    bmask = maskobj.array

    fp = source.getFootprint()
    peak = fp.getPeaks()[0]

    # this is a Point2D but at an integer location
    peak_location = peak.getCentroid()

    jacob = _extract_jacobian(
        subim=subim,
        source=source,
        orig_cen=peak_location,
    )

    psf_im = _extract_psf_image(exposure=subim, orig_cen=peak_location)

    # fake the psf pixel noise
    psf_err = psf_im.max()*0.0001
    psf_wt = psf_im*0 + 1.0/psf_err**2

    # use canonical center for the psf
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0
    psf_jacob = jacob.copy()
    psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

    # we will have need of the bit names which we can only
    # get from the mask object
    # this is sort of monkey patching, but I'm not sure of
    # a better solution
    meta = {
        'maskobj': maskobj,
        'orig_cen': peak_location,
    }

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacob,
    )
    obs = ngmix.Observation(
        im,
        weight=wt,
        bmask=bmask,
        jacobian=jacob,
        psf=psf_obs,
        meta=meta,
    )

    return obs


def _extract_jacobian(subim, source, orig_cen):
    """
    extract an ngmix.Jacobian from the image object
    and object record

    Parameters
    ----------
    imobj: lsst.afw.image.Exposure
        An Exposure object, e.g. ExposureF
    source: lsst.afw.table.SourceRecord
        A source record from detection/deblending
    orig_cen: Point2D
        Location of object in original image

    returns
    --------
    Jacobian: ngmix.Jacobian
        The local jacobian
    """

    # this will still be the wcs for the full exposure
    wcs = subim.getWcs()

    xy0 = subim.getXY0()
    stamp_cen = orig_cen - geom.Extent2D(xy0)

    LOG.debug('cen in subim: %s', stamp_cen)
    row = stamp_cen.getY()
    col = stamp_cen.getX()

    # we get this at the original center
    linear_wcs = wcs.linearizePixelToSky(
        orig_cen,  # loc in original image
        geom.arcseconds,
    )
    jmatrix = linear_wcs.getLinear().getMatrix()

    jacob = ngmix.Jacobian(
        row=row,
        col=col,
        dudcol=jmatrix[0, 0],
        dudrow=jmatrix[0, 1],
        dvdcol=jmatrix[1, 0],
        dvdrow=jmatrix[1, 1],
    )

    return jacob


def _extract_weight(subim):
    """
    TODO get the estimated sky variance rather than this hack
    TODO should we zero out other bits?

    extract a weight map

    Areas with NO_DATA will get zero weight.

    Because the variance map includes the full poisson variance, which
    depends on the signal, we instead extract the median of the parts of
    the image without NO_DATA set

    parameters
    ----------
    subim: sub exposure object
    """

    # TODO implement bit checking
    var_image = subim.variance.array

    weight = var_image.copy()

    weight[:, :] = 0

    wuse = np.where(var_image > 0)

    if wuse[0].size > 0:
        medvar = np.median(var_image[wuse])
        weight[:, :] = 1.0/medvar
    else:
        print('    weight is all zero, found '
              'none that passed cuts')

    return weight


def _extract_psf_image(exposure, orig_cen):
    """
    get the psf associated with this image.

    coadded psfs from DM are generally not square, but the coadd in cells code
    makes them so.  We will assert they are square and odd dimensions
    """
    try:
        psfobj = exposure.getPsf()
        psfim = psfobj.computeKernelImage(orig_cen).array
    except InvalidParameterError:
        raise MissingDataError("could not reconstruct PSF")

    psfim = np.array(psfim, dtype='f4', copy=False)

    shape = psfim.shape
    assert shape[0] == shape[1], 'require square psf images'
    assert shape[0] % 2 != 0, 'require odd psf images'

    return psfim


def find_and_set_center(obs, rng, ntry=4, fwhm=1.2):
    """
    Attempt to find the centroid and update the jacobian.  Update
    'orig_cen' in the metadata with the difference. Add entry
    "orig_cen_offset" as an Extend2D

    If the centroiding fails, raise CentroidFail
    """

    obs.meta['orig_cen_offset'] = geom.Extent2D(x=np.nan, y=np.nan)

    res = ngmix.admom.find_cen_admom(obs, fwhm=fwhm, rng=rng, ntry=ntry)
    if res['flags'] != 0:
        raise CentroidFail('failed to find centroid')

    jac = obs.jacobian

    # this is an offset in arcsec
    voff, uoff = res['cen']

    # current center within stamp, in pixels
    rowcen, colcen = jac.get_cen()

    # new center within stamp, in pixels
    new_row, new_col = jac.get_rowcol(u=uoff, v=voff)

    # difference, which we will use to update the center in the original image
    rowdiff = new_row - rowcen
    coldiff = new_col - colcen

    diff = geom.Extent2D(x=coldiff, y=rowdiff)

    obs.meta['orig_cen'] = obs.meta['orig_cen'] + diff
    obs.meta['orig_cen_offset'] = diff

    # update jacobian center within the stamp
    with obs.writeable():
        obs.jacobian.set_cen(row=new_row, col=new_col)


def get_output(obs, wcs, fitter, source, res, pres, ormask, stamp_size, exp_bbox):
    """
    get the output structure, copying in results

    When data are unavailable, a default value of nan is used
    """
    meas_type = get_meas_type(fitter)
    output = get_output_struct(meas_type)

    n = util.Namer(front=meas_type)

    output['psf_flags'] = pres['flags']
    output[n('flags')] = res['flags']

    output['stamp_size'] = stamp_size
    output['row0'] = exp_bbox.getBeginY()
    output['col0'] = exp_bbox.getBeginX()

    if obs is not None:
        orig_cen = obs.meta['orig_cen']
        cen_offset = obs.meta['orig_cen_offset']
        output['row'] = orig_cen.getY()
        output['col'] = orig_cen.getX()

        output['row_diff'] = cen_offset.getY()
        output['col_diff'] = cen_offset.getX()

        skypos = wcs.pixelToSky(orig_cen)

        output['ra'] = skypos.getRa().asDegrees()
        output['dec'] = skypos.getDec().asDegrees()

    output['ormask'] = ormask

    flags = 0
    if pres['flags'] != 0 and pres['flags'] != procflags.NO_ATTEMPT:
        flags |= procflags.PSF_FAILURE

    if res['flags'] != 0:
        flags |= res['flags'] | procflags.OBJ_FAILURE

    if pres['flags'] == 0:
        output['psf_g'] = pres['g']
        output['psf_T'] = pres['T']

    if 'T' in res:
        output[n('T')] = res['T']
        output[n('T_err')] = res['T_err']

    if 'flux' in res:
        output[n('flux')] = res['flux']
        output[n('flux_err')] = res['flux_err']

    if res['flags'] == 0:
        output[n('s2n')] = res['s2n']
        output[n('g')] = res['g']
        output[n('g_cov')] = res['g_cov']

        if pres['flags'] == 0:
            output[n('T_ratio')] = res['T']/pres['T']

    output['flags'] = flags
    return output


class CentroidFail(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)
