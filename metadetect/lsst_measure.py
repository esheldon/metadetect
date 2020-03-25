import numpy as np
import ngmix
import esutil as eu

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

import lsst.log

from . import util
from . import procflags
from .lsst_mbobs_extractor import MBObsMissingDataError


def measure_weighted_moments(*, mbobs, weight, thresh=10, loglevel='INFO'):
    assert len(mbobs) == 1, 'one combined band for now'
    assert len(mbobs[0]) == 1, 'one epoch only'

    exposure = mbobs[0][0].exposure

    sources, meas_task = detect_and_deblend(
        exposure=exposure,
        thresh=thresh,
        loglevel=loglevel,
    )

    replacer = _get_noise_replacer(exposure=exposure, sources=sources)

    results = []

    for source in sources:

        # Skip parent objects where all children are inserted
        if source.get('deblend_nChild') != 0:
            continue

        # This will insert a single source into the image
        replacer.insertSource(source.getId())

        meas_task.callMeasure(source, exposure)

        # TODO variable box size
        bbox = _get_bbox_fixed(
            exposure=exposure,
            source=source,
            stamp_size=48,
        )
        subim = _get_padded_sub_image(exposure=exposure, bbox=bbox)

        obs = _extract_obs(
            subim=subim,
            source=source,
        )

        pres = _measure_moments(obs=obs.psf, weight=weight)
        ores = _measure_moments(obs=obs, weight=weight)

        res = _get_output(source=source, res=ores, pres=pres)

        # Remove object
        replacer.removeSource(source.getId())

        results.append(res)

    # Insert all objects back into image
    replacer.end()

    if len(results) > 0:
        results = eu.numpy_util.combine_arrlist(results)
    else:
        results = None

    return results


def detect_and_deblend(*, exposure, thresh=10, loglevel='INFO'):

    loglevel = loglevel.upper()

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

    # setup detection config
    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = thresh
    detection_task = SourceDetectionTask(config=detection_config)
    detection_task.log.setLevel(getattr(lsst.log, loglevel))

    deblend_config = SourceDeblendConfig()
    deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)
    deblend_task.log.setLevel(getattr(lsst.log, loglevel))

    # Detect objects
    table = afw_table.SourceTable.make(schema)
    result = detection_task.run(table, exposure)
    sources = result.sources

    # run the deblender
    deblend_task.run(exposure, sources)

    return sources, meas_task


def _get_noise_replacer(*, exposure, sources):
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
    )


def _extract_obs(*, subim, source):
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
    obs = ngmix.Observation(
        im,
        weight=wt,
        bmask=bmask,
        jacobian=jacob,
        psf=psf_obs,
        meta=meta,
    )

    return obs


def _get_bbox_fixed(*, exposure, source, stamp_size):
    radius = stamp_size/2
    radius = int(np.ceil(radius))

    bbox = _project_box(
        source=source,
        wcs=exposure.getWcs(),
        radius=radius,
    )
    return bbox


def _get_bbox_calc(*,
                   exposure,
                   source,
                   min_stamp_size,
                   max_stamp_size,
                   sigma_factor):
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
    # except LogicError as err:
    except LogicError:
        bbox = source.getFootprint().getBBox()

    return bbox


def _compute_stamp_size(*,
                        source,
                        min_stamp_size,
                        max_stamp_size,
                        sigma_factor):
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


def _project_box(*, source, wcs, radius):
    """
    create a box for the input source
    """
    pixel = geom.Point2I(wcs.skyToPixel(source.getCoord()))
    box = geom.Box2I()
    box.include(pixel)
    box.grow(radius)
    return box


def _get_padded_sub_image(*, exposure, bbox):
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


def _extract_psf_image(*, exposure, orig_cen):
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
        raise MBObsMissingDataError("could not reconstruct PSF")

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

    """
    # TODO set the ignor bits
    bitnames_to_ignore = self.config['stamps']['bits_to_ignore_for_weight']

    bits_to_ignore = util.get_ored_bits(maskobj, bitnames_to_ignore)

    maskobj = subim.mask
    mask = maskobj.array
    wuse = np.where(
        (var_image > 0) &
        ((mask & bits_to_ignore) == 0)
    )

    if wuse[0].size > 0:
        medvar = np.median(var_image[wuse])
        weight[:, :] = 1.0/medvar
    else:
        self.log.debug('    weight is all zero, found '
                       'none that passed cuts')
        # _print_bits(maskobj, bitnames_to_ignore)

    bitnames_to_null = self.config['stamps']['bits_to_null']
    if len(bitnames_to_null) > 0:
        bits_to_null = util.get_ored_bits(maskobj, bitnames_to_null)
        wnull = np.where((mask & bits_to_null) != 0)
        if wnull[0].size > 0:
            self.log.debug('    nulling %d in weight' % wnull[0].size)
            weight[wnull] = 0.0

    return weight
    """


def _extract_jacobian(*, subim, source):
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


def _measure_moments(*, obs, weight):
    res = weight.get_weighted_moments(obs=obs, maxrad=1.e9)

    if res['flags'] != 0:
        return res

    res['numiter'] = 1
    res['g'] = res['e']
    res['g_cov'] = res['e_cov']

    return res


def _get_dtype():

    model = 'wmom'
    npars = 6
    n = util.Namer(front=model)
    dt = [
        ('flags', 'i4'),

        ('row', 'f4'),
        ('col', 'f4'),

        ('psfrec_flags', 'i4'),  # psfrec is the original psf
        ('psfrec_g', 'f8', 2),
        ('psfrec_T', 'f8'),

        ('psf_flags', 'i4'),
        ('psf_g', 'f8', 2),
        ('psf_T', 'f8'),

        (n('flags'), 'i4'),
        (n('s2n'), 'f8'),
        (n('pars'), 'f8', npars),
        (n('g'), 'f8', 2),
        (n('g_cov'), 'f8', (2, 2)),
        (n('T'), 'f8'),
        (n('T_err'), 'f8'),
        (n('T_ratio'), 'f8'),
    ]

    return dt


def _get_output(*, source, res, pres):

    model = 'wmom'
    n = util.Namer(front=model)

    dt = _get_dtype()
    output = np.zeros(1, dtype=dt)

    output['psfrec_flags'] = procflags.NO_ATTEMPT

    output['psf_flags'] = pres['flags']
    output[n('flags')] = res['flags']

    orig_cen = source.getCentroid()

    if np.isnan(orig_cen.getY()):
        peak = source.getFootprint().getPeaks()[0]
        orig_cen = peak.getI()

    output['row'] = orig_cen.getY()
    output['col'] = orig_cen.getX()

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
        output[n('pars')] = res['pars']
        output[n('g')] = res['g']
        output[n('g_cov')] = res['g_cov']
        output[n('T')] = res['T']
        output[n('T_err')] = res['T_err']

        if pres['flags'] == 0:
            output[n('T_ratio')] = res['T']/pres['T']

    output['flags'] = flags
    return output
