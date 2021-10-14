import numpy as np
import ngmix

import lsst.afw.table as afw_table
import lsst.afw.image as afw_image
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
)
import lsst.geom as geom
from lsst.pex.exceptions import (
    InvalidParameterError,
    LogicError,
)

from . import util
from . import vis
from . import procflags
from .defaults import DEFAULT_THRESH


def detect_and_deblend(exposure, thresh=DEFAULT_THRESH):
    """
    run detection and deblending of peaks.  The SDSS deblender is run in order
    to split peaks, but need not be used to create deblended stamps.

    we must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    exposure: Exposure
        The exposure to process
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

    # this must occur directly before any tasks are run because schema is
    # modified in place by tasks, and the constructor does a check that
    # fails if we construct it separately

    deblend_task = SourceDeblendTask(
        config=SourceDeblendConfig(),
        schema=schema,
    )

    table = afw_table.SourceTable.make(schema)
    result = detection_task.run(table, exposure)

    if result is not None:
        sources = result.sources
        deblend_task.run(exposure, sources)
    else:
        sources = []

    return sources, meas_task


def measure(
    exposure,
    sources,
    fitter,
    stamp_size,
    meas_task=None,
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
    """

    if len(sources) == 0:
        return None

    exp_bbox = exposure.getBBox()
    results = []

    # ormasks will be different within the loop below due to the replacer
    ormasks = get_ormasks(sources=sources, exposure=exposure)

    for i, source in enumerate(sources):

        # Skip parent objects where all children are inserted
        if source.get('deblend_nChild') != 0:
            continue

        ormask = ormasks[i]

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
            vis.show_exp(subim)

        obs = _extract_obs(exp=subim, source=source)
        if obs is None:
            # all zero weights for the image this occurs when we have zeros in
            # the weight plane near the edge but the image is non-zero. These
            # are always junk
            psf_res = {'flags': procflags.NO_ATTEMPT}
            object_res = {'flags': procflags.ZERO_WEIGHTS}
            stamp_size = -1
        else:
            psf_res = measure_one(obs=obs.psf, fitter=fitter)
            object_res = measure_one(obs=obs, fitter=fitter)
            stamp_size = obs.image.shape[0]

        res = get_output(
            wcs=exposure.getWcs(), fitter=fitter, source=source, res=object_res,
            psf_res=psf_res, ormask=ormask, stamp_size=stamp_size, exp_bbox=exp_bbox,
        )

        results.append(res)

    if len(results) > 0:
        results = np.hstack(results)
    else:
        results = None

    return results


def measure_one(obs, fitter):
    """
    run a measurement on an input observation

    Parameters
    ----------
    obs: ngmix.Observation
        The observation to measure
    fitter: e.g. ngmix.prepsfmom.PrePsfGaussMom
        The measurement object

    Returns
    -------
    res dict
    """
    from ngmix.ksigmamom import KSigmaMom
    from ngmix.prepsfmom import PrePSFGaussMom

    is_prepsf = (
        isinstance(fitter, KSigmaMom) or isinstance(fitter, PrePSFGaussMom)
    )
    if is_prepsf and not obs.has_psf():
        res = fitter.go(obs, no_psf=True)
    else:
        res = fitter.go(obs)

    if res['flags'] != 0:
        return res

    res['numiter'] = 1
    res['g'] = res['e']
    res['g_cov'] = res['e_cov']

    if isinstance(fitter, ngmix.runners.Runner):
        assert isinstance(fitter.fitter, ngmix.admom.AdmomFitter)
        gm = res.get_gmix()
        obs.set_gmix(gm)
        psf_flux_fitter = ngmix.fitting.PSFFluxFitter(do_psf=False)
        flux_res = psf_flux_fitter.go(obs)
        res['flux'] = flux_res['flux']
        res['flux_err'] = flux_res['flux_err']

        # gm.set_flux(res['flux'])
        # im = gm.make_image(obs.image.shape, jacobian=obs.jacobian)
        # from espy import images
        # images.compare_images(obs.image, im)

    return res


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
    ormasks = []
    for source in sources:
        ormask = get_ormask(source=source, exposure=exposure)
        ormasks.append(ormask)
    return ormasks


def get_ormask(source, exposure):
    """
    get ormask based on original peak position

    Parameters
    ----------
    sources: lsst.afw.table.SourceRecord
        The sources
    exposure: lsst.afw.image.ExposureF
        The exposure

    Returns
    -------
    ormask value
    """
    peak = source.getFootprint().getPeaks()[0]
    orig_cen = peak.getI()
    maskval = exposure.mask[orig_cen]
    return maskval


def _extract_obs(exp, source):
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
        The Observation unless all the weight are zero, in which
        case None is returned
    """

    im = exp.image.array
    # im = im - _get_bg_from_edges(image=im, border=2)

    wt = _extract_weight(exp)
    if np.all(wt <= 0):
        return None

    maskobj = exp.mask
    bmask = maskobj.array
    jacob = _extract_jacobian(
        exp=exp,
        source=source,
    )

    # TODO using fixed kernel for now
    orig_cen = source.getCentroid()
    # orig_cen = exp.getWcs().skyToPixel(source.getCoord())

    psf_im = _extract_psf_image(exposure=exp, orig_cen=orig_cen)

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
    obs = ngmix.Observation(
        im,
        weight=wt,
        bmask=bmask,
        jacobian=jacob,
        psf=psf_obs,
        meta=meta,
    )

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
        result.setPsf(exposure.getPsf().clone())

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


def _extract_weight(exp):
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
    exp: sub exposure object
    """

    # TODO implement bit checking
    var_image = exp.variance.array

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


def _extract_jacobian(exp, source):
    """
    extract an ngmix.Jacobian from the image object
    and object record

    exp: an image object
        TODO I don't actually know what class this is
    source: an object record
        TODO I don't actually know what class this is

    returns
    --------
    Jacobian: ngmix.Jacobian
        The local jacobian
    """

    xy0 = exp.getXY0()

    orig_cen = exp.getWcs().skyToPixel(source.getCoord())

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

    wcs = exp.getWcs().linearizePixelToSky(
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


def _get_dtype(meas_type, nband):

    n = util.Namer(front=meas_type)
    dt = [
        ('flags', 'i4'),

        ('stamp_size', 'i4'),
        ('row0', 'i4'),  # bbox row start
        ('col0', 'i4'),  # bbox col start
        ('row', 'f4'),  # row in image. Use row0 to get to global pixel coords
        ('col', 'f4'),  # col in image. Use col0 to get to global pixel coords
        ('row_diff', 'f4'),  # difference from peak location
        ('col_diff', 'f4'),  # difference from peak location
        ('row_noshear', 'f4'),  # noshear row in local image, not global wcs
        ('col_noshear', 'f4'),  # noshear col in local image, not global wcs
        ('ra', 'f8'),
        ('dec', 'f8'),

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
        # (n('flux'), 'f8'),
        # (n('flux_err'), 'f8'),
    ]

    if nband > 1:
        dt += [
            (n('band_flux'), 'f8', nband),
            (n('band_flux_err'), 'f8', nband),
        ]
    else:
        dt += [
            (n('band_flux'), 'f8'),
            (n('band_flux_err'), 'f8'),
        ]

    return dt


def get_output_struct(meas_type, nband=1):
    n = util.Namer(front=meas_type)
    dt = _get_dtype(meas_type, nband=nband)

    output = np.zeros(1, dtype=dt)

    output['flags'] = procflags.NO_ATTEMPT
    output['psf_flags'] = procflags.NO_ATTEMPT

    output[n('s2n')] = np.nan
    output[n('g')] = np.nan
    output[n('T')] = np.nan
    output[n('T_err')] = np.nan
    output[n('g_cov')] = np.nan
    output[n('T_ratio')] = np.nan
    output[n('band_flux')] = np.nan
    output[n('band_flux_err')] = np.nan

    output['psf_g'] = np.nan
    output['psf_T'] = np.nan

    return output


def get_output(wcs, fitter, source, res, psf_res, ormask, stamp_size, exp_bbox):
    """
    get the output structure, copying in results

    When data are unavailable, a default value of nan is used
    """
    meas_type = get_meas_type(fitter)

    if 'band_flux' in res:
        nband = len(res['band_flux'])
    else:
        nband = 1

    output = get_output_struct(meas_type, nband=nband)

    n = util.Namer(front=meas_type)

    output['psf_flags'] = psf_res['flags']
    output[n('flags')] = res['flags']

    orig_cen = source.getCentroid()

    skypos = wcs.pixelToSky(orig_cen)

    if np.isnan(orig_cen.getY()):
        peak = source.getFootprint().getPeaks()[0]
        orig_cen = peak.getI()

    output['stamp_size'] = stamp_size
    output['row0'] = exp_bbox.getBeginY()
    output['col0'] = exp_bbox.getBeginX()
    output['row'] = orig_cen.getY()
    output['col'] = orig_cen.getX()

    output['ra'] = skypos.getRa().asDegrees()
    output['dec'] = skypos.getDec().asDegrees()

    output['ormask'] = ormask

    flags = 0
    if psf_res['flags'] != 0 and psf_res['flags'] != procflags.NO_ATTEMPT:
        flags |= procflags.PSF_FAILURE

    if res['flags'] != 0:
        flags |= res['flags'] | procflags.OBJ_FAILURE

    if psf_res['flags'] == 0:
        output['psf_g'] = psf_res['g']
        output['psf_T'] = psf_res['T']

    if 'T' in res:
        output[n('T')] = res['T']
        output[n('T_err')] = res['T_err']

    if 'band_flux' in res:
        output[n('band_flux')] = res['band_flux']
        output[n('band_flux_err')] = res['band_flux_err']

    if res['flags'] == 0:
        output[n('s2n')] = res['s2n']
        output[n('g')] = res['g']
        output[n('g_cov')] = res['g_cov']

        if psf_res['flags'] == 0:
            output[n('T_ratio')] = res['T']/psf_res['T']

    output['flags'] = flags
    return output


def get_meas_type(fitter):
    """
    Get the measurement type as a string for the input fitter

    Parameters
    ----------
    fitter: e.g. ngmix.prepsfmom.PrePSFGaussMom
        The fitter

    Returns
    -------
    name for measurement type
    """
    if isinstance(fitter, ngmix.gaussmom.GaussMom):
        meas_type = 'wmom'
    elif isinstance(fitter, ngmix.ksigmamom.KSigmaMom):
        meas_type = 'ksigma'
    elif isinstance(fitter, ngmix.prepsfmom.PrePSFGaussMom):
        meas_type = 'gap'
    elif isinstance(fitter, ngmix.runners.Runner):
        assert isinstance(fitter.fitter, ngmix.admom.AdmomFitter), (
            'only meas_type "am" for a runner'
        )
        meas_type = 'am'
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
