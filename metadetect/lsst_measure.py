import numpy as np
import ngmix

import lsst.afw.table as afw_table
import lsst.afw.image as afw_image
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
    NoiseReplacerConfig,
    NoiseReplacer,
)
import lsst.geom as geom
from lsst.pex.exceptions import (
    InvalidParameterError,
    LogicError,
)
from .lsst_skysub import determine_and_subtract_sky

import logging

from . import util
from . import procflags
from .defaults import (
    DEFAULT_LOGLEVEL,
    DEFAULT_THRESH,
    DEFAULT_DEBLEND,
)


def detect_deblend_and_measure(
    exposure,
    fitter,
    stamp_size,
    thresh=DEFAULT_THRESH,
    deblend=DEFAULT_DEBLEND,
    noise_image=None,
    loglevel=DEFAULT_LOGLEVEL,
):
    """
    run detection, deblending and measurements.

    Note deblending is always run in a hierarchical detection process, but the
    user has a choice whether to use deblended postage stamps for the
    measurement.

    Parameters
    ----------
    exposure: Exposure
        The exposure on which to detect and measure
    fitter: e.g. ngmix.gaussmom.GaussMom or ngmix.ksigmamom.KSigmaMom
        For calculating moments
    thresh: float
        The detection threshold in units of the sky noise
    stamp_size: int
        Size for postage stamps.
    deblend: bool
        If True, run the deblender.
    noise_image: array
        A noise image for use by the NoiseReplacer.  If you are running
        metacal you should send the same image for all metacal images.
    loglevel: str, optional
        Log level for logger in string form
    """

    sources, meas_task = detect_and_deblend(
        exposure=exposure,
        thresh=thresh,
        loglevel=loglevel,
    )

    return measure(
        exposure=exposure,
        sources=sources,
        fitter=fitter,
        meas_task=meas_task,
        stamp_size=stamp_size,
        deblend=deblend,
        noise_image=noise_image,
    )


def measure(
    exposure,
    sources,
    fitter,
    stamp_size,
    meas_task=None,
    deblend=DEFAULT_DEBLEND,
    noise_image=None,
):
    """
    run measurements on the input exposure, given the input measurement task,
    list of sources, and fitter.  These steps are combined because of the way
    that the deblending works to produce noise-replaced images for neighbors
    where the image is temporarily modified.

    To avoid data inconsistency in the case an exception is raised, a copy of
    the exposure is made when using the noise replacer.

    Parameters
    ----------
    exposure: Exposure
        The exposure on which to detect and measure
    sources: list of sources
        From a detection task
    fitter: e.g. ngmix.gaussmom.GaussMom or ngmix.ksigmamom.KSigmaMom
        For calculating moments
    stamp_size: int
        Size for postage stamps
    meas_task: SingleFrameMeasurementTask
        An optional measurement task; if you already have centeroids etc. for
        sources, no need to send it.  Otherwise this should do basic things
        like finding the centroid
    deblend: bool
        If True, deblend neighbors.
    noise_image: array
        A noise image for use by the NoiseReplacer.  If you are running
        metacal you should send the same image for all metacal images.
    """

    if len(sources) > 0:
        if deblend:
            # this makes a copy of everything, including pixels
            exp_send = afw_image.ExposureF(exposure, deep=True)
        else:
            exp_send = exposure

        results = _do_measure(
            exposure=exp_send,
            sources=sources,
            fitter=fitter,
            stamp_size=stamp_size,
            meas_task=meas_task,
            deblend=deblend,
            noise_image=noise_image,
        )
    else:
        results = None

    return results


def _do_measure(
    exposure,
    sources,
    fitter,
    stamp_size,
    meas_task,
    deblend,
    noise_image=None,
    seed=None,
):
    """
    See docs for measure()
    """

    if deblend:
        # remove all objects and replace with noise
        noise_replacer = _get_noise_replacer(
            exposure=exposure, sources=sources, noise_image=noise_image,
        )
    else:
        noise_replacer = None

    exp_bbox = exposure.getBBox()
    results = []

    # ormasks will be different within the loop below due to the replacer
    ormasks = _get_ormasks(sources=sources, exposure=exposure)

    for i, source in enumerate(sources):

        # Skip parent objects where all children are inserted
        if source.get('deblend_nChild') != 0:
            continue

        ormask = ormasks[i]

        if deblend:
            # This will insert a single source into the image
            noise_replacer.insertSource(source.getId())

        if meas_task is not None:
            # results are stored in the source
            meas_task.callMeasure(source, exposure)

        # TODO variable box size?
        stamp_bbox = _get_bbox_fixed(
            exposure=exposure,
            source=source,
            stamp_size=stamp_size,
        )
        subim = _get_padded_sub_image(exposure=exposure, bbox=stamp_bbox)
        if False:
            show_exp(subim)

        obs = _extract_obs(subim=subim, source=source)
        if obs is None:
            # we hit an edge or some blank area, this is always junk
            continue

        pres = _measure_one(obs=obs.psf, fitter=fitter)
        ores = _measure_one(obs=obs, fitter=fitter)

        res = _get_output(
            fitter=fitter, source=source, res=ores, pres=pres, ormask=ormask,
            box_size=obs.image.shape[0], exp_bbox=exp_bbox,
        )

        if deblend:
            # Remove object
            noise_replacer.removeSource(source.getId())

        results.append(res)

    if deblend:
        # put exposure back as it was input
        noise_replacer.end()

    if len(results) > 0:
        results = np.hstack(results)
    else:
        results = None

    return results


def show_exp(exp):
    """
    show the image in ds9

    Parameters
    ----------
    exp: Exposure
        The image to show
    """
    import lsst.afw.display as afw_display
    display = afw_display.getDisplay(backend='ds9')
    display.mtv(exp)
    display.scale('log', 'minmax')

    input('hit a key')


def _measure_one(obs, fitter):
    """
    run a measurement on an input observation
    """
    from ngmix.ksigmamom import KSigmaMom

    if isinstance(fitter, KSigmaMom) and not obs.has_psf():
        res = fitter.go(obs, no_psf=True)
    else:
        res = fitter.go(obs)

    if res['flags'] != 0:
        return res

    res['numiter'] = 1
    res['g'] = res['e']
    res['g_cov'] = res['e_cov']

    return res


def _get_ormasks(*, sources, exposure):
    """
    get a list of all the ormasks for the sources
    """
    ormasks = []
    for source in sources:
        ormask = _get_ormask(source=source, exposure=exposure)
        ormasks.append(ormask)
    return ormasks


def _get_ormask(*, source, exposure):
    """
    get ormask based on original peak position
    """
    peak = source.getFootprint().getPeaks()[0]
    orig_cen = peak.getI()
    maskval = exposure.mask[orig_cen]
    return maskval


def detect_and_deblend(
    exposure,
    thresh=DEFAULT_THRESH,
    loglevel=DEFAULT_LOGLEVEL,
):
    """
    run detection and deblending of peaks

    we must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    exposure: Exposure
        The exposure to process
    thresh: float, optional
        The detection threshold in units of the sky noise
    loglevel: str, optional
        Log level for logger in string form

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
        # "base_SdssShape",
        # "base_LocalBackground",
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
    detection_task.log.setLevel(getattr(logging, loglevel.upper()))

    # this must occur directly before any tasks are run because schema is
    # modified in place by tasks, and the constructor does a check that
    # fails if we construct it separately
    deblend_config = SourceDeblendConfig()

    deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)
    deblend_task.log.setLevel(getattr(logging, loglevel.upper()))

    table = afw_table.SourceTable.make(schema)
    result = detection_task.run(table, exposure)

    if result is not None:
        sources = result.sources
        deblend_task.run(exposure, sources)
    else:
        sources = []

    return sources, meas_task


def iterate_detection_and_skysub(
    exposure, thresh, niter=2, loglevel=DEFAULT_LOGLEVEL,
):
    """
    Iterate detection and sky subtraction

    Parameters
    ----------
    exposure: Exposure
        The exposure to process
    thresh: float
        threshold for detection
    niter: int, optional
        Number of iterations for detection and sky subtraction.
        Must be >= 1. Default is 2 which is recommended.

    Returns
    -------
    Result from running the detection task
    """
    from lsst.pex.exceptions import RuntimeError as LSSTRuntimeError
    from lsst.pipe.base.task import TaskError
    if niter < 1:
        raise ValueError(f'niter {niter} is less than 1')

    schema = afw_table.SourceTable.makeMinimalSchema()
    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = thresh
    detection_task = SourceDetectionTask(config=detection_config)
    detection_task.log.setLevel(getattr(logging, loglevel.upper()))

    table = afw_table.SourceTable.make(schema)

    # keep a running sum of each sky that was subtracted
    try:
        sky_meas = 0.0
        for i in range(niter):
            determine_and_subtract_sky(exposure)
            result = detection_task.run(table, exposure)

            sky_meas += exposure.getMetadata()['BGMEAN']

        meta = exposure.getMetadata()

        # this is the overall sky we subtracted in all iterations
        meta['BGMEAN'] = sky_meas
    except LSSTRuntimeError as err:
        err = str(err).replace('lsst::pex::exceptions::RuntimeError:', '')
        detection_task.log.warn(err)
        result = None
    except TaskError as err:
        err = str(err).replace('lsst.pipe.base.task.TaskError:', '')
        detection_task.log.warn(err)
        result = None

    return result


def subtract_sky_mbobs(mbobs, thresh):
    """
    subtract sky

    We combine these because both involve resetting the image
    and noise image

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to sky subtract
    thresh: float
        Threshold for detection
    """
    for obslist in mbobs:
        for obs in obslist:
            exp = obs.coadd_exp

            _ = iterate_detection_and_skysub(
                exposure=exp,
                thresh=thresh,
            )

            obs.image = exp.image.array


def _get_noise_replacer(exposure, sources, noise_image=None):
    """
    get a noise replacer for the input exposure and source list
    """

    # Notes for metacal.
    #
    # For metacal we should generate a noise image so that the exact noise
    # field is used for all versions of the metacal images.  The assumption is
    # that, because these noise data should contain no signal, metacal is not
    # calibrating it.  Thus it doesn't matter whether or not the noise field is
    # representative of the full covariance of the true image noise.  Rather by
    # making the field the same for all metacal images we reduce variance in
    # the calculation of the response

    noise_replacer_config = NoiseReplacerConfig()
    footprints = {
        source.getId(): (source.getParent(), source.getFootprint())
        for source in sources
    }

    # This constructor will replace all detected pixels with noise in the
    # image
    return NoiseReplacer(
        noise_replacer_config,
        exposure=exposure,
        footprints=footprints,
        noiseImage=noise_image,
    )


def _extract_obs(subim, source):
    """
    convert an image object into an ngmix.Observation, including
    a psf observation

    parameters
    ----------
    imobj: an image object
        TODO I don't actually know what class this is
    source: an object sourceord
        TODO I don't actually know what class this is

    returns
    --------
    obs: ngmix.Observation
        The Observation, including
    """

    im = subim.image.array
    # im = im - _get_bg_from_edges(image=im, border=2)

    wt = _extract_weight(subim)
    maskobj = subim.mask
    bmask = maskobj.array
    jacob = _extract_jacobian(
        subim=subim,
        source=source,
    )

    # TODO using fixed kernel for now
    orig_cen = source.getCentroid()
    # orig_cen = subim.getWcs().skyToPixel(source.getCoord())

    psf_im = _extract_psf_image(exposure=subim, orig_cen=orig_cen)

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
    meta = {'maskobj': maskobj}

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacob,
    )
    try:
        obs = ngmix.Observation(
            im,
            weight=wt,
            bmask=bmask,
            jacobian=jacob,
            psf=psf_obs,
            meta=meta,
        )
    except ngmix.GMixFatalError:
        print('skipping junk object with all zero weight')
        obs = None

    return obs


def _get_bbox_fixed(exposure, source, stamp_size):
    """
    get a fixed sized bounding box
    """
    radius = stamp_size/2
    radius = int(np.ceil(radius))

    bbox = _project_box(
        source=source,
        wcs=exposure.getWcs(),
        radius=radius,
    )
    return bbox


def _get_bbox_calc(
    exposure,
    source,
    min_stamp_size,
    max_stamp_size,
    sigma_factor,
):
    """
    get a bounding box with size based on measurements
    """
    try:
        stamp_radius, stamp_size = _compute_stamp_size(
            source=source,
            min_stamp_size=min_stamp_size,
            max_stamp_size=max_stamp_size,
            sigma_factor=sigma_factor,
        )
        bbox = _project_box(
            source=source,
            wcs=exposure.getWcs(),
            radius=stamp_radius,
        )
    except LogicError:
        bbox = source.getFootprint().getBBox()

    return bbox


def _compute_stamp_size(
    source,
    min_stamp_size,
    max_stamp_size,
    sigma_factor,
):
    """
    calculate a stamp radius for the input object, to
    be used for constructing postage stamp sizes
    """

    min_radius = min_stamp_size/2
    max_radius = max_stamp_size/2

    quad = source.getShape()
    T = quad.getIxx() + quad.getIyy()  # noqa
    if np.isnan(T):  # noqa
        T = 4.0  # noqa

    sigma = np.sqrt(T/2.0)  # noqa
    radius = sigma_factor*sigma

    if radius < min_radius:
        radius = min_radius
    elif radius > max_radius:
        radius = max_radius

    radius = int(np.ceil(radius))
    stamp_size = 2*radius+1

    return radius, stamp_size


def _project_box(source, wcs, radius):
    """
    create a box for the input source
    """
    pixel = geom.Point2I(wcs.skyToPixel(source.getCoord()))
    box = geom.Box2I()
    box.include(pixel)
    box.grow(radius)
    return box


def _get_padded_sub_image(exposure, bbox):
    """
    extract a sub-image, padded out when it is not contained
    """
    region = exposure.getBBox()

    if region.contains(bbox):
        return exposure.Factory(exposure, bbox, afw_image.PARENT, True)

    result = exposure.Factory(bbox)
    bbox2 = geom.Box2I(bbox)
    bbox2.clip(region)

    if isinstance(exposure, afw_image.Exposure):
        result.setPsf(exposure.getPsf())
        result.setWcs(exposure.getWcs())
        result.setPhotoCalib(exposure.getPhotoCalib())
        # result.image.array[:, :] = float("nan")
        result.image.array[:, :] = 0.0
        result.variance.array[:, :] = float("inf")
        result.mask.array[:, :] = \
            np.uint16(result.mask.getPlaneBitMask("NO_DATA"))
        sub_in = afw_image.MaskedImageF(
            exposure.maskedImage, bbox=bbox2,
            origin=afw_image.PARENT, deep=False
        )
        result.maskedImage.assign(sub_in, bbox=bbox2, origin=afw_image.PARENT)

    elif isinstance(exposure, afw_image.ImageI):
        result.array[:, :] = 0
        sub_in = afw_image.ImageI(exposure, bbox=bbox2,
                                  origin=afw_image.PARENT, deep=False)
        result.assign(sub_in, bbox=bbox2, origin=afw_image.PARENT)

    else:
        raise ValueError("Image type not supported")

    return result


def _extract_psf_image(exposure, orig_cen):
    """
    get the psf associated with this image

    coadded psfs are generally not square, so we will
    trim it to be square and preserve the center to
    be at the new canonical center

    TODO: should we really trim the psf to be even?  will this
    cause a shift due being off-center?
    """
    try:
        psfobj = exposure.getPsf()
        psfim = psfobj.computeKernelImage(orig_cen).array
    except InvalidParameterError:
        raise MissingDataError("could not reconstruct PSF")

    psfim = np.array(psfim, dtype='f4', copy=False)

    psfim = util.trim_odd_image(psfim)
    return psfim


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


def _extract_jacobian(subim, source):
    """
    extract an ngmix.Jacobian from the image object
    and object record

    subim: an image object
        TODO I don't actually know what class this is
    source: an object record
        TODO I don't actually know what class this is

    returns
    --------
    Jacobian: ngmix.Jacobian
        The local jacobian
    """

    xy0 = subim.getXY0()

    orig_cen = subim.getWcs().skyToPixel(source.getCoord())

    if np.isnan(orig_cen.getY()):
        print('falling back on integer location')
        # fall back to integer pixel location
        peak = source.getFootprint().getPeaks()[0]
        orig_cen_i = peak.getI()
        orig_cen = geom.Point2D(
            x=orig_cen_i.getX(),
            y=orig_cen_i.getY(),
        )
        # x, y = peak.getIx(), peak.getIy()

    cen = orig_cen - geom.Extent2D(xy0)
    row = cen.getY()
    col = cen.getX()

    wcs = subim.getWcs().linearizePixelToSky(
        orig_cen,
        geom.arcseconds,
    )
    jmatrix = wcs.getLinear().getMatrix()

    jacob = ngmix.Jacobian(
        row=row,
        col=col,
        dudcol=jmatrix[0, 0],
        dudrow=jmatrix[0, 1],
        dvdcol=jmatrix[1, 0],
        dvdrow=jmatrix[1, 1],
    )

    return jacob


def _get_dtype(meas_type):

    n = util.Namer(front=meas_type)
    dt = [
        ('flags', 'i4'),

        ('box_size', 'i4'),
        ('row0', 'i4'),  # bbox row start
        ('col0', 'i4'),  # bbox col start
        ('row', 'f4'),  # row in image. Use row0 to get to global pixel coords
        ('col', 'f4'),  # col in image. Use col0 to get to global pixel coords
        ('row_noshear', 'f4'),  # noshear row in local image, not global wcs
        ('col_noshear', 'f4'),  # noshear col in local image, not global wcs

        ('psfrec_flags', 'i4'),  # psfrec is the original psf
        ('psfrec_g', 'f8', 2),
        ('psfrec_T', 'f8'),

        ('psf_flags', 'i4'),
        ('psf_g', 'f8', 2),
        ('psf_T', 'f8'),

        ('ormask', 'i4'),
        ('mfrac', 'f4'),

        (n('flags'), 'i4'),
        (n('s2n'), 'f8'),
        (n('g'), 'f8', 2),
        (n('g_cov'), 'f8', (2, 2)),
        (n('T'), 'f8'),
        (n('T_err'), 'f8'),
        (n('T_ratio'), 'f8'),
        (n('flux'), 'f8'),
        (n('flux_err'), 'f8'),
    ]

    return dt


def _get_output(fitter, source, res, pres, ormask, box_size, exp_bbox):

    meas_type = _get_meas_type(fitter)
    n = util.Namer(front=meas_type)

    dt = _get_dtype(meas_type)
    output = np.zeros(1, dtype=dt)

    output['psfrec_flags'] = procflags.NO_ATTEMPT

    output['psf_flags'] = pres['flags']
    output[n('flags')] = res['flags']

    orig_cen = source.getCentroid()

    if np.isnan(orig_cen.getY()):
        peak = source.getFootprint().getPeaks()[0]
        orig_cen = peak.getI()

    output['box_size'] = box_size
    output['row0'] = exp_bbox.getBeginY()
    output['col0'] = exp_bbox.getBeginX()
    output['row'] = orig_cen.getY() - output['row0']
    output['col'] = orig_cen.getX() - output['col0']
    output['ormask'] = ormask

    flags = 0
    if pres['flags'] != 0:
        flags |= procflags.PSF_FAILURE

    if res['flags'] != 0:
        flags |= procflags.OBJ_FAILURE

    if pres['flags'] == 0:
        output['psf_g'] = pres['g']
        output['psf_T'] = pres['T']

    if res['flags'] == 0:
        output[n('s2n')] = res['s2n']
        output[n('g')] = res['g']
        output[n('g_cov')] = res['g_cov']
        output[n('T')] = res['T']
        output[n('T_err')] = res['T_err']
        output[n('flux')] = res['flux']
        output[n('flux_err')] = res['flux_err']

        if pres['flags'] == 0:
            output[n('T_ratio')] = res['T']/pres['T']

    output['flags'] = flags
    return output


def _get_meas_type(fitter):
    if isinstance(fitter, ngmix.gaussmom.GaussMom):
        meas_type = 'wmom'
    elif isinstance(fitter, ngmix.ksigmamom.KSigmaMom):
        meas_type = 'ksigma'
    else:
        raise ValueError(
            "don't know how to get name for fitter %s" % repr(fitter)
        )

    return meas_type


class MissingDataError(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)
