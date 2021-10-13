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
    get_output,
    get_ormask,
    _get_bbox_fixed,
    _get_padded_sub_image,
    _extract_obs,
)
from .lsst_measure_scarlet import measure_one, extract_obs
import logging

LOG = logging.getLogger('lsst_measure_shredder')


def detect_and_deblend(
    mbexp, rng,
    fitter, stamp_size, thresh=DEFAULT_THRESH,
):
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

    Tvals = {}
    if result is not None:
        sources = result.sources
        deblend_task.run(detexp, sources)

        with ContextNoiseReplacer(detexp, sources, rng) as replacer:

            for source in sources:
                with replacer.sourceInserted(source.getId()):
                    meas_task.callMeasure(source, detexp)

                    stamp_bbox = _get_bbox_fixed(
                        exposure=detexp,
                        source=source,
                        stamp_size=stamp_size,
                    )

                    subim = _get_padded_sub_image(exposure=detexp, bbox=stamp_bbox)

                    obs = _extract_obs(subim=subim, source=source)
                    if obs is None:
                        T = 0.3
                    else:
                        ores = measure_one(obs=obs, fitter=fitter)
                        T = ores['T']

                    Tvals[source.getId()] = T

    else:
        sources = []

    return sources, detexp, Tvals


def measure(
    mbexp,
    detexp,
    sources,
    fitter,
    stamp_size,
    Tvals,
    rng,
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

    wcs = detexp.getWcs()
    exp_bbox = detexp.getBBox()
    ormasks = get_ormasks(sources=sources, exposure=detexp)

    if show:
        vis.show_mbexp(mbexp, mess='Original')

    with MultibandNoiseReplacer(mbexp, sources, rng) as replacer:

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
                # if show or parent_id == 17:
                if show:
                    vis.show_mbexp(replacer.mbexp, mess=f'{parent_id} inserted')

                children = sources.getChildren(parent_id)
                nchild = len(children)

                try:
                    if nchild == 0:
                        LOG.info('no children, processing parent')
                        parent_mbexp = get_stamp(
                            replacer.mbexp, parent, stamp_size=stamp_size,
                        )

                        res = _process_parent(
                            parent_mbexp=parent_mbexp, stamp_size=stamp_size,
                            source=parent,
                            fitter=fitter, wcs=wcs, exp_bbox=exp_bbox,
                            ormask=ormasks[parent.getId()],
                            rng=rng, show=show,
                        )
                        these_results = [res]
                    else:
                        LOG.info(f'deblending {nchild} child objects')

                        bbox = get_blend_bbox(
                            exp=replacer.mbexp, sources=children,
                            stamp_size=stamp_size,
                            grow_footprint=10,  # 5 on each side
                        )
                        blend_mbexp = get_stamp(replacer.mbexp, parent, bbox=bbox)
                        # if show or parent_id == 17:
                        if show:
                            vis.show_mbexp(blend_mbexp, mess=f'{parent_id} stamp')

                        these_results = _process_blend(
                            blend_mbexp=blend_mbexp, children=children,
                            fitter=fitter, wcs=wcs, exp_bbox=exp_bbox,
                            ormasks=ormasks,
                            rng=rng,
                            stamp_size=stamp_size,
                            Tvals=Tvals, show=show,
                        )

                    results += these_results

                except LengthError as err:
                    LOG.info('failed to get bbox: %s', err)
                    # note the context manager properly handles a return
                    ores = {'flags': procflags.BBOX_HITS_EDGE}
                    pres = {'flags': procflags.NO_ATTEMPT}

                    if nchild == 0:
                        tosend = [parent]
                    else:
                        tosend = children

                    # fill out the now labeled failures
                    results += [
                        get_output(wcs=wcs, fitter=fitter,
                                   source=source, res=ores, pres=pres,
                                   ormask=ormasks[source.getId()],
                                   stamp_size=stamp_size,
                                   exp_bbox=exp_bbox)
                        for source in tosend
                    ]

    if len(results) > 0:
        results = np.hstack(results)
    else:
        results = None

    return results


def _process_parent(
    parent_mbexp, stamp_size, source, fitter, wcs, rng, ormask, exp_bbox,
    show=False,
):
    coadded_stamp_exp = util.coadd_exposures(parent_mbexp.singles)
    obs = extract_obs(subim=coadded_stamp_exp, source=source)

    if obs is None:
        LOG.info('skipping object with all zero weights')
        ores = {'flags': procflags.ZERO_WEIGHTS}
        pres = {'flags': procflags.NO_ATTEMPT}
    else:
        pres = measure_one(obs=obs.psf, fitter=fitter)
        ores = measure_one(obs=obs, fitter=fitter)

    return get_output(
        wcs=wcs, fitter=fitter,
        source=source, res=ores, pres=pres,
        ormask=ormask,
        stamp_size=stamp_size,
        exp_bbox=exp_bbox,
    )


def _process_blend(
    blend_mbexp, children, wcs, rng, Tvals, stamp_size, fitter, ormasks, exp_bbox,
    show=False,
):
    from shredder import ModelSubtractor
    from shredder.coadding import make_coadd_obs

    nchild = len(children)

    # Use center of footprint bbox for reconstructing the psf and
    # the calculating the jacobian
    orig_cen = children[0].getFootprint().getBBox().getCenter()

    shredder = make_shredder(
        mbexp=blend_mbexp, orig_cen=orig_cen, rng=rng,
        psf_ngauss=3,
    )
    guess = get_shredder_guess(
        shredder=shredder,
        sources=children,
        Tvals=Tvals,
        bbox=blend_mbexp.singles[0].getBBox(),
        init_model='exp',
        rng=rng,
    )

    shredder.shred(guess)
    assert shredder.result['flags'] == 0

    if show:
        vis.compare_mbexp(blend_mbexp, shredder.get_model_images())

    subtractor = ModelSubtractor(shredder, nchild)

    results = []
    for ichild, child in enumerate(children):
        with subtractor.add_source(ichild):
            stamp_mbobs = subtractor.get_object_mbobs(
                index=ichild, stamp_size=stamp_size,
            )
            # if show:
            #     subtractor.plot_object(
            #         index=ichild, stamp_size=stamp_size,
            #     )

            # TODO work multi-band
            coadd_stamp_mbobs = make_coadd_obs(stamp_mbobs)

            pres = measure_one(obs=coadd_stamp_mbobs.psf, fitter=fitter)
            ores = measure_one(obs=coadd_stamp_mbobs, fitter=fitter)

            res = get_output(
                wcs=wcs, fitter=fitter, source=child,
                res=ores, pres=pres,
                ormask=ormasks[child.getId()],
                stamp_size=stamp_size, exp_bbox=exp_bbox,
            )
            results.append(res)

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


def get_blend_bbox(exp, sources, stamp_size, grow_footprint=None):
    """
    get a bbox for the blend.  Start with the footprint and grow as
    needed to support the requested stamp size
    """
    # this is a copy
    bbox = sources[0].getFootprint().getBBox()
    if grow_footprint is not None:
        bbox.grow(grow_footprint // 2)

    for i, source in enumerate(sources):
        this_bbox = get_bbox(
            exp=exp, source=source, stamp_size=stamp_size,
        )
        bbox.include(this_bbox)

    return bbox


def get_bbox(
    exp, source, stamp_size=None, clip=False,
):
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
        bbox = source.getFootprint().getBBox()

    return bbox


def _extract_jacobian_for_shredding(wcs, orig_cen):
    """
    extract an ngmix.Jacobian with row0, col0 at 0, 0 from the exposure

    Parameters
    ----------
    wcs: WCS object
        The wcs object for calculating the jacobian
    orig_cen: Point2D
        Location of object in original image

    returns
    --------
    Jacobian: ngmix.Jacobian
        The local jacobian
    """

    # we get this at the original center
    linear_wcs = wcs.linearizePixelToSky(
        orig_cen,  # loc in original image
        geom.arcseconds,
    )
    jmatrix = linear_wcs.getLinear().getMatrix()

    jacob = ngmix.Jacobian(
        row=0,
        col=0,
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


def make_shredder(mbexp, orig_cen, rng, psf_ngauss=5):
    """
    Create a Shredder instance from the input MultibandExposure and sources

    Requires converting the mbexp to an ngmix.MultiBandObsList, which results
    in some data copying

    for the observations we set the origin of the jacobian to 0, 0 to simplify
    translation from the coord system of the exp

    TODO make psf_ngauss configurable
    TODO make optional args to the Shredder configurable
    """
    from shredder import Shredder

    mbobs = _extract_mbobs_for_shredding(mbexp=mbexp, orig_cen=orig_cen)
    return Shredder(
        obs=mbobs,
        psf_ngauss=psf_ngauss,
        # miniter=100,
        # flux_miniter=40,
        rng=rng,
        # tol=1.0e-6,
    )


def _extract_mbobs_for_shredding(mbexp, orig_cen):
    mbobs = ngmix.MultiBandObsList()

    jacobian = _extract_jacobian_for_shredding(
        wcs=mbexp.singles[0].getWcs(), orig_cen=orig_cen,
    )

    for exp in mbexp.singles:
        obs = _extract_obs_for_shredding(
            exp=exp, jacobian=jacobian, orig_cen=orig_cen,
        )

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def _extract_obs_for_shredding(exp, jacobian, orig_cen):
    """
    convert an exposure object into an ngmix.Observation, including
    a psf observation.

    Parameters
    ----------
    exp: lsst.afw.image.Exposure
        An Exposure object, e.g. ExposureF
    jacobian: ngmix.Jacobian
        The jacobian.  Should have origin at 0, 0

    returns
    --------
    obs: ngmix.Observation
        The Observation unless all the weight are zero, in which
        case None is returned
    """

    im = exp.image.array

    wt = _extract_weight(exp)
    if np.all(wt <= 0):
        return None

    maskobj = exp.mask
    bmask = maskobj.array

    psf_im = _extract_psf_image(exposure=exp, orig_cen=orig_cen)

    # fake the psf pixel noise
    psf_err = psf_im.max()*0.0001
    psf_wt = psf_im*0 + 1.0/psf_err**2

    # use canonical center for the psf
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0
    psf_jacob = jacobian.copy()
    psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

    # we will have need of the bit names which we can only
    # get from the mask object
    # this is sort of monkey patching, but I'm not sure of
    # a better solution
    meta = {
        'maskobj': maskobj,
        'orig_cen': orig_cen,
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
        jacobian=jacobian,
        psf=psf_obs,
        meta=meta,
    )

    return obs


def get_shredder_guess(
    shredder, sources, Tvals, bbox, rng, minflux=0.01, init_model='dev',
):
    """
    get a guess for the shredder.  Currently we have no guesses for size on
    input, but we do have psf fluxes.  For now guess a multiple of the psf
    size;  in the future we will get T guesses from measurements on the SDSS
    deblended stamps

    TODO make minflux configurable
    """

    ur = rng.uniform

    obs = shredder.mbobs[0][0]
    Tpsf = obs.psf.gmix.get_T()
    jacobian = obs.jacobian

    corner = bbox.getMin()

    scale = jacobian.scale

    guess_pars = []

    psf_T = shredder.mbobs[0][0].psf.gmix.get_T()
    Tmin = 0.5 * psf_T
    for i, source in enumerate(sources):

        Tguess = Tpsf*ur(low=1.0, high=1.4)
        Tguess = Tvals[source.getId()]
        if Tguess < Tmin:
            Tguess = Tmin

        # location in big bounding box for exposure
        cen = source.getCentroid()
        flux = source.getPsfInstFlux()

        if flux < minflux:
            LOG.debug('flux %g less than minflux %g', flux, minflux)
            flux = minflux

        # convert to local coords in arcsec
        v, u = jacobian.get_vu(
            row=cen.y - corner.y,
            col=cen.x - corner.x,
        )
        LOG.debug('v: %g u: %g', v, u)

        g1, g2 = ur(low=-0.01, high=0.01, size=2)

        pars = [v, u, g1, g2, Tguess, flux]
        gm_model = ngmix.GMixModel(pars, init_model)

        LOG.debug('gm model guess')
        LOG.debug('\n%s', gm_model)

        # perturb the models to avoid degeneracies
        data = gm_model.get_data()
        for j in range(data.size):
            data['p'][j] *= ur(low=0.95, high=1.05)

            fac = 0.01
            data['row'][j] += ur(low=-fac*scale, high=fac*scale)
            data['col'][j] += ur(low=-fac*scale, high=fac*scale)

            data['irr'][j] *= ur(low=0.95, high=1.05)
            data['irc'][j] *= ur(low=0.95, high=1.05)
            data['icc'][j] *= ur(low=0.95, high=1.05)

        guess_pars += list(gm_model.get_full_pars())

    gm_guess = ngmix.GMix(pars=guess_pars)
    return gm_guess


def get_ormasks(sources, exposure):
    """
    get a list of all the ormasks for the sources

    Parameters
    ----------
    sources: lsst.afw.table.SourceCatalog
        The sources
    exposure: lsst.afw.image.ExposureF
        The exposure

    Returns
    -------
    list of ormask values
    """
    ormasks = {}
    for source in sources:
        ormasks[source.getId()] = get_ormask(source=source, exposure=exposure)
    return ormasks
