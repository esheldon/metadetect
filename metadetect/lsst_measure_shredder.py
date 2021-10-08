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
from copy import deepcopy
import ngmix
import numpy as np
import lsst.afw.table as afw_table
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
)
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
import lsst.geom as geom

from lsst.pex.exceptions import InvalidParameterError, LengthError
from . import vis
from . import util
from .util import MultibandNoiseReplacer, ContextNoiseReplacer
from .defaults import DEFAULT_THRESH
from . import procflags
from .lsst_measure import (
    get_ormask,
    measure_one,
)
from .lsst_measure_scarlet import get_output
import logging

LOG = logging.getLogger('lsst_measure_shredder')


def detect_and_deblend(mbexp, thresh=DEFAULT_THRESH):
    """
    run detection and deblending of peaks, as well as basic measurments such as
    centroid.  The SDSS deblender is run in order to split peaks.

    We must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    thresh: float, optional
        The detection threshold in units of the sky noise

    Returns
    -------
    sources, meas_task
        The sources and the measurement task
    """

    schema = afw_table.SourceTable.makeMinimalSchema()

    # Setup algorithms to run
    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_PsfFlux",
        "base_SkyCoord",
    ]

    # set these slots to none because we aren't running these algorithms
    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None

    # goes with SdssShape above
    meas_config.slots.shape = None

    # fix odd issue where it things things are near the edge
    meas_config.plugins['base_SdssCentroid'].binmax = 1

    meas_task = SingleFrameMeasurementTask(
        config=meas_config,
        schema=schema,
    )

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = thresh
    detection_task = SourceDetectionTask(
        # TODO should we send schema?
        config=detection_config,
    )

    # this must occur directly before any tasks are run because schema is
    # modified in place by tasks, and the constructor does a check that
    # fails if we construct it separately

    deblend_task = SourceDeblendTask(
        config=SourceDeblendConfig(),
        schema=schema,
    )

    table = afw_table.SourceTable.make(schema)

    if len(mbexp.singles) > 1:
        detexp = util.coadd_exposures(mbexp.singles)
    else:
        detexp = mbexp.singles[0]

    result = detection_task.run(table, detexp)

    if result is not None:
        sources = result.sources
        deblend_task.run(detexp, sources)

        with ContextNoiseReplacer(detexp, sources) as replacer:

            for source in sources:
                with replacer.sourceInserted(source.getId()):
                    meas_task.callMeasure(source, detexp)

    else:
        sources = []

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
        The detection exposure, used for getting ormasks
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

    with MultibandNoiseReplacer(mbexp=mbexp, sources=sources) as replacer:

        if show:
            vis.show_mbexp(replacer.mbexp, mess='All replaced')

        results = []

        parents = sources.getChildren(0)

        LOG.info('found %d parents', len(parents))

        for parent in parents:

            parent_id = parent.getId()

            LOG.info('-'*70)
            LOG.info('parent id: %d', parent_id)

            with replacer.sourceInserted(parent_id):
                if show:
                    vis.show_mbexp(replacer.mbexp, mess=f'{parent_id} inserted')

                children = sources.getChildren(parent_id)
                nchild = len(children)

                if nchild == 0:
                    LOG.info('no children, processing parent')
                    cen = parent.getCentroid()
                    LOG.info('parent centroid: %s', cen)
                    stamp = get_stamp(replacer.mbexp, parent, stamp_size=stamp_size)
                else:
                    LOG.info(f'processing {nchild} child objects')
                    # we need to deblend it
                    for child in children:
                        cen = child.getCentroid()
                        LOG.info('child centroid: %s', cen)

                    bbox = get_blend_bbox(
                        exp=replacer.mbexp, sources=children,
                        stamp_size=stamp_size,
                    )
                    stamp = get_stamp(replacer.mbexp, parent, bbox=bbox)

                if show:
                    vis.show_mbexp(stamp, mess=f'{parent_id} stamp')

        # for source in sources:
        #
        #     # we want to insert a parent, which inserts the original pixels
        #     if source.get('deblend_nChild') == 0:
        #         continue
        #
        #     parent_id = source.getId()
        #
        #     LOG.info('-'*70)
        #     LOG.info('parent id: %d', parent_id)
        #
        #     with replacer.sourceInserted(parent_id):
        #         if show:
        #             vis.show_mbexp(replacer.mbexp, mess=f'{parent_id} inserted')
        #
        #         # not sending stamp_size, getting the footprint bounding box
        #         stamp = get_stamp(replacer.mbexp, source)
        #         if show:
        #             vis.show_mbexp(stamp, mess=f'{parent_id} stamp')
        #
        #         children = sources.getChildren(parent_id)
        #         nchild = len(children)
        #         LOG.info(
        #             'processing %d %s',
        #             nchild, 'children' if nchild > 1 else 'child'
        #         )
        #         for child in children:
        #             cen = child.getCentroid()
        #             LOG.info('child centroid: %s', cen)
        #         stop
        #
        #         results += _process_sources(
        #             subtractor=subtractor, sources=children, stamp_size=stamp_size,
        #             detexp=detexp, wcs=wcs, fitter=fitter, rng=rng, show=show,
        #         )

    if len(results) > 0:
        results = np.hstack(results)
    else:
        results = None

    return results


def get_stamp(mbexp, source, stamp_size=None, clip=False, bbox=None):
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

    Returns
    -------
    ExposureF
    """

    if bbox is None:
        bbox = get_bbox(exp=mbexp, source=source, stamp_size=stamp_size, clip=clip)

    exposures = [mbexp[band][bbox] for band in mbexp.filters]
    return util.get_mbexp(exposures)


def get_blend_bbox(exp, sources, stamp_size):
    """
    get a bbox for the blend.  Start with the footprint and grow as
    needed to support the requested stamp size
    """
    bbox = deepcopy(sources[0].getFootprint().getBBox())

    for i, source in enumerate(sources):
        this_bbox = get_bbox(exp=exp, source=source, stamp_size=stamp_size)
        bbox.include(this_bbox)

    return bbox


def get_bbox(exp, source, stamp_size=None, clip=False):
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

    fp = source.getFootprint()

    if stamp_size is not None:
        cen = source.getCentroid()

        bbox = geom.Box2I(
            geom.Point2I(cen),
            geom.Extent2I(1, 1),
        )
        bbox.grow(stamp_size // 2)

        exp_bbox = exp.getBBox()
        if clip:
            bbox.clip(exp_bbox)
        else:
            if not exp_bbox.contains(bbox):
                source_id = source.getId()
                raise LengthError(
                    f'requested stamp size {stamp_size} for source '
                    f'{source_id} does not fit into the exposoure.  '
                    f'Use clip=True to clip the bbox to fit'
                )

    else:
        bbox = fp.getBBox()

    return bbox


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

    box_size = -1
    with subtractor.add_source(source_id):

        if show:
            vis.show_mbexp(
                subtractor.mbexp, mess=f'source {source_id} added'
            )

        try:
            stamp_mbexp = subtractor.get_stamp(
                source_id, stamp_size=stamp_size,
            )

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
            obs = _extract_obs(wcs=wcs, subim=coadded_stamp_exp, source=source)
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
                    box_size = obs.image.shape[0]

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

        res = get_output(
            obs=obs, wcs=wcs, fitter=fitter, source=source, res=ores, pres=pres,
            ormask=ormask, box_size=box_size, exp_bbox=exp_bbox,
        )

    return res


def _extract_obs(wcs, subim, source):
    """
    convert an exposure object into an ngmix.Observation, including
    a psf observation.

    the center for the jacobian will be at the peak location, an integer
    location.

    Parameters
    ----------
    wcs: stack wcs
        The wcs for the full image not this sub image
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
        wcs=wcs,
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


def _extract_jacobian(wcs, subim, source, orig_cen):
    """
    extract an ngmix.Jacobian from the image object
    and object record

    Parameters
    ----------
    wcs: stack wcs
        The wcs for the full image not this sub image
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


class MissingDataError(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


class CentroidFail(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)
