import logging
import warnings
import numpy as np
import esutil as eu
import ngmix

import lsst.afw.table as afw_table
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.algorithms import (
    SourceDetectionConfig as OriginalSourceDetectionConfig,
)
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
)
from lsst.pex.config import Config, ConfigurableField, Field
from lsst.pipe.base import Task
import lsst.afw.image as afw_image
import lsst.geom as geom
from lsst.pex.exceptions import (
    InvalidParameterError,
    LengthError,
)

from ..procflags import (
    EDGE_HIT,
    ZERO_WEIGHTS,
    NO_ATTEMPT,
)
from ..fitting import (
    fit_mbobs_gauss,
    fit_mbobs_wavg,
    get_wavg_output_struct,
)

from . import util
from .util import ContextNoiseReplacer, get_jacobian
from . import vis
from .defaults import DEFAULT_THRESH

warnings.filterwarnings('ignore', category=FutureWarning)

LOG = logging.getLogger('lsst_measure')


class SourceDetectionConfig(OriginalSourceDetectionConfig):
    @property
    def thresh(self):
        return self.thresholdValue

    @thresh.setter
    def thresh(self, value):
        self.thresholdValue = value


SourceDetectionTask.ConfigClass = SourceDetectionConfig


class DetectAndDeblendConfig(Config):
    meas = ConfigurableField[SingleFrameMeasurementConfig](
        doc="Measurement config",
        target=SingleFrameMeasurementTask,
    )

    detect = ConfigurableField[SourceDetectionConfig](
        doc="Detection config", target=SourceDetectionTask
    )

    deblend = ConfigurableField[SourceDeblendConfig](
        doc="Deblend config", target=SourceDeblendTask
    )

    seed = Field[int](
        doc="Random rng seed",
        default=42,
    )

    def setDefaults(self):
        super().setDefaults()

        # defaults for measurement config
        self.meas.plugins.names = [
            "base_SdssCentroid",
            "base_PsfFlux",
            "base_SkyCoord",
        ]

        # set these slots to none because we aren't running these algorithms
        self.meas.slots.apFlux = None
        self.meas.slots.gaussianFlux = None
        self.meas.slots.calibFlux = None
        self.meas.slots.modelFlux = None

        # goes with SdssShape above
        self.meas.slots.shape = None

        # fix odd issue where it things things are near the edge
        self.meas.plugins['base_SdssCentroid'].binmax = 1

        # defaults for detection config
        # DM does not have config default stability.  Set all of them
        # explicitly
        self.detect.minPixels = 1
        self.detect.isotropicGrow = True
        self.detect.combinedGrow = True
        self.detect.nSigmaToGrow = 2.4
        self.detect.returnOriginalFootprints = False
        self.detect.includeThresholdMultiplier = 1.0
        self.detect.thresholdPolarity = "positive"
        self.detect.adjustBackground = 0.0
        self.detect.doApplyFlatBackgroundRatio = False
        # these are ignored since we are doing reEstimateBackground = False
        # detection_config.background
        # detection_config.tempLocalBackground
        # detection_config.doTempLocalBackground
        # detection_config.tempWideBackground
        # detection_config.doTempWideBackground
        self.detect.nPeaksMaxSimple = 1
        self.detect.nSigmaForKernel = 7.0
        self.detect.excludeMaskPlanes = util.get_detection_mask()

        # the defaults changed from from stdev to pixel_std but
        # we don't want that

        self.detect.thresholdType = "stdev"
        # our changes from defaults
        self.detect.reEstimateBackground = False

        self.detect.thresholdValue = DEFAULT_THRESH

        # these will be ignored when finding the image standard deviation
        self.detect.statsMask = util.get_stats_mask()

        # deblend config
        # these tasks must use the same schema and all be constructed before
        # any other tasks using the same schema are run because schema is
        # modified in place by tasks, and the constructor does a check that
        # fails if we do this afterward
        self.deblend.maxFootprintArea = 0


class DetectAndDeblendTask(Task):
    ConfigClass = DetectAndDeblendConfig
    _DefaultName = "detect_and_deblend"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        schema = afw_table.SourceTable.makeMinimalSchema()
        self.makeSubtask("meas", schema=schema)
        self.makeSubtask("detect", schema=schema)
        self.makeSubtask("deblend", schema=schema)
        self.rng = np.random.RandomState(seed=self.config.seed)

    def run(self, mbexp, show=False):
        if len(mbexp.singles) > 1:
            detexp = util.coadd_exposures(mbexp.singles)
            if detexp is None:
                return [], None
        else:
            detexp = mbexp.singles[0]

        # background measurement within the detection code requires ExposureF
        if not isinstance(detexp, afw_image.ExposureF):
            detexp = afw_image.ExposureF(detexp, deep=True)

        schema = self.deblend.schema  # should be the same for all tasks
        table = afw_table.SourceTable.make(schema)
        result = self.detect.run(table, detexp)

        if result is not None:
            sources = result.sources
            self.deblend.run(detexp, sources)

            with ContextNoiseReplacer(
                detexp,
                sources,
                self.rng,
                config=self.meas.config.noiseReplacer,
            ) as replacer:
                for source in sources:
                    if source.get('deblend_nChild') != 0:
                        continue

                    source_id = source.getId()

                    with replacer.sourceInserted(source_id):
                        self.meas.callMeasure(source, detexp)

        else:
            sources = []

        if show:
            vis.show_exp(detexp, use_mpl=True, sources=sources)

        return sources, detexp


def detect_and_deblend(
    mbexp,
    rng=None,
    thresh=DEFAULT_THRESH,
    show=False,
    config=None,
):
    """
    run detection and deblending of peaks, as well as basic measurments such as
    centroid.  The SDSS deblender is run in order to split footprints.

    We must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    rng: np.random.RandomState
        Random number generator for noise replacer
    thresh: float, optional
        The detection threshold in units of the sky noise
    show: bool, optional
        If set to True, show images

    Returns
    -------
    sources, detexp
        The sources and the detection exposure
    """
    config_override = config if config is not None else {}
    if thresh:
        if 'detect' not in config_override:
            config_override['detect'] = {}
        config_override['detect']['thresholdValue'] = thresh

    config = DetectAndDeblendConfig()
    config.setDefaults()

    util.override_config(config, config_override)

    config.freeze()
    config.validate()
    task = DetectAndDeblendTask(config=config)
    if rng is not None:
        task.rng = rng
    return task.run(mbexp, show)


def measure(
    mbexp,
    detexp,
    sources,
    config,
    rng,
):
    """
    run measurements on the input exposure, given the input measurement task,
    list of sources, and config.

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposure on which to detect and measure
    detexp: lsst.afw.image.Exposure*
        The detection exposure, used for bmask info
    sources: list of sources
        From a detection task
    config: dict, optional
        Configuration for the fitter, metacal, psf, detect, Entries
        in this dict override defaults; see lsst_configs.py
    rng: np.random.RandomState
        Random number generator

    Returns
    -------
    ndarray with results or None
    """

    if len(sources) == 0:
        return None

    pgauss_fitter = get_pgauss_fitter(config)

    nband = len(mbexp.bands)
    shear_band_names = config["shear_bands"] or mbexp.bands
    if not all([sb in mbexp.bands for sb in shear_band_names]):
        raise RuntimeError(
            "Not all requested bands for shear are available. "
            f"Bands `{shear_band_names}` were requested but the only "
            f"bands available are `{mbexp.bands}`."
        )
    shear_bands = [
        i for i, band in enumerate(mbexp.bands) if band in shear_band_names
    ]
    exp_bbox = mbexp.getBBox()
    wcs = mbexp.singles[0].getWcs()
    results = []

    # bmasks will be different within the loop below due to the replacer
    bmasks = get_bmasks(sources=sources, exposure=detexp)

    for i, source in enumerate(sources):
        if source.get('deblend_nChild') != 0:
            continue

        bmask = bmasks[i]

        stamp_flags = 0
        try:
            mbobs = _get_stamp_mbobs(
                mbexp=mbexp,
                source=source,
                stamp_size=config['stamp_size'],
            )
        except LengthError as err:
            # This is raised when a bbox hits an edge
            LOG.debug('%s', err)
            stamp_flags = EDGE_HIT
        except AllZeroWeightError as err:
            # failure creating some observation due to all weights being zero
            # across the stamp
            LOG.info('%s', err)
            stamp_flags = ZERO_WEIGHTS

        if stamp_flags != 0:

            this_gauss_res = get_wavg_output_struct(
                nband=nband,
                model='gauss',
                shear_bands=shear_bands,
            )
            this_pgauss_res = get_wavg_output_struct(
                nband=nband,
                model='pgauss',
            )

        else:

            # TODO do something with bmask_flags?
            psf_fitter = ngmix.admom.AdmomFitter(rng=rng)
            psf_guesser = ngmix.guessers.GMixPSFGuesser(
                rng=rng,
                ngauss=1,
                guess_from_moms=True,
            )
            psf_runner = ngmix.runners.PSFRunner(
                fitter=psf_fitter,
                guesser=psf_guesser,
                ntry=4,
            )

            this_gauss_res = fit_mbobs_gauss(
                mbobs=mbobs,
                bmask_flags=0,
                psf_runner=psf_runner,
                rng=rng,
                shear_bands=shear_bands,
            )

            this_pgauss_res = fit_mbobs_wavg(
                mbobs=mbobs,
                fitter=pgauss_fitter,
                bmask_flags=0,
            )

        this_res = _get_combined_struct(this_gauss_res, this_pgauss_res)
        this_res['stamp_flags'] = stamp_flags

        res = get_output(
            wcs=wcs,
            source=source,
            res=this_res,
            bmask=bmask,
            stamp_size=config['stamp_size'],
            exp_bbox=exp_bbox,
        )

        results.append(res)

    if len(results) > 0:
        results = eu.numpy_util.combine_arrlist(results)
    else:
        results = None

    return results


def get_pgauss_fitter(config):
    return ngmix.prepsfmom.PGaussMom(fwhm=config['pgauss']['fwhm'])


def _get_combined_struct(gauss_res, pgauss_res):

    skip = ['pgauss_g', 'pgauss_g_cov', 'shear_bands']
    keep_dt = [('stamp_flags', 'i4')]

    for pdt in pgauss_res.dtype.descr:

        n = pdt[0]

        if 'psf' in n or n in skip:
            continue

        keep_dt.append(pdt)

    out = eu.numpy_util.add_fields(gauss_res, keep_dt)

    for n in pgauss_res.dtype.names:
        if n in out.dtype.names and n != "shear_bands":
            out[n] = pgauss_res[n]

    return out


def get_bmasks(sources, exposure):
    """
    get a list of all the bmasks for the sources

    Parameters
    ----------
    sources: lsst.afw.table.SourceCatalog
        The sources
    exposure: lsst.afw.image.ExposureF
        The exposure

    Returns
    -------
    list of bmask values
    """
    bmasks = []
    for source in sources:
        bmask = get_bmask(source=source, exposure=exposure)
        bmasks.append(bmask)
    return bmasks


def get_bmask(source, exposure):
    """
    get bmask based on original peak position

    Parameters
    ----------
    sources: lsst.afw.table.SourceRecord
        The sources
    exposure: lsst.afw.image.ExposureF
        The exposure

    Returns
    -------
    bmask value
    """
    peak = source.getFootprint().getPeaks()[0]
    orig_cen = peak.getI()
    maskval = exposure.mask[orig_cen]
    return maskval


def extract_obs(exp, source):
    """
    convert an image object into an ngmix.Observation, including
    a psf observation

    parameters
    ----------
    imobj: lsst.afw.image.ExposureF
        The exposure
    source: lsst.afw.table.SourceRecord
        The source record

    returns
    --------
    obs: ngmix.Observation
        The Observation unless all the weight are zero, in which
        case AllZeroWeightError is raised
    """

    im = exp.image.array

    wt = _extract_weight(exp)
    if np.all(wt <= 0):
        raise AllZeroWeightError('all weights <= 0')

    bmask = exp.mask.array
    jacob = _extract_jacobian_at_source(
        exp=exp,
        source=source,
    )

    orig_cen = source.getCentroid()

    psf_im = extract_psf_image(exposure=exp, orig_cen=orig_cen)

    # fake the psf pixel noise
    psf_err = psf_im.max() * 0.0001
    psf_wt = psf_im * 0 + 1.0 / psf_err**2

    # use canonical center for the psf
    psf_cen = (np.array(psf_im.shape) - 1.0) / 2.0
    psf_jacob = jacob.copy()
    psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

    # we will have need of the bit names which we can only
    # get from the mask object
    # this is sort of monkey patching, but I'm not sure of
    # a better solution

    meta = {'orig_cen': orig_cen}

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


def _get_stamp_mbobs(mbexp, source, stamp_size, clip=False):
    """
    Get a postage stamp MultibandExposure

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures
    source: lsst.afw.table.SourceRecord
        The source for which to get the stamp
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
    lsst.afw.image.ExposureF
    """

    bbox = _get_bbox(mbexp, source, stamp_size, clip=clip)

    mbobs = ngmix.MultiBandObsList()
    for band in mbexp.bands:
        subexp = mbexp[band][bbox]
        obs = extract_obs(
            exp=subexp,
            source=source,
        )

        obslist = ngmix.ObsList(meta={'band': band})
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def _get_bbox(mbexp, source, stamp_size, clip=False):
    """
    Get a bounding box at the location of the specified source.

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures
    source: lsst.afw.table.SourceRecord
        The source for which to get the stamp
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
    peak = fp.getPeaks()[0]

    x_peak, y_peak = peak.getIx(), peak.getIy()

    bbox = geom.Box2I(
        geom.Point2I(x_peak, y_peak),
        geom.Extent2I(1, 1),
    )
    bbox.grow(stamp_size // 2)

    exp_bbox = mbexp.getBBox()
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

    return bbox


def extract_psf_image(exposure, orig_cen):
    """
    get the psf associated with this image.

    coadded psfs from DM are generally not square, but the coadd in cells code
    makes them so.  We will assert they are square and odd dimensions

    Parameters
    ----------
    exposure: lsst.afw.image.ExposureF
        The exposure data
    orig_cen: lsst.geom.Point2D
        The location at which to draw the image

    Returns
    -------
    ndarray
    """
    try:
        psfobj = exposure.getPsf()
        psfim = psfobj.computeKernelImage(orig_cen).array
    except InvalidParameterError:
        raise MissingDataError("could not reconstruct PSF")

    psfim = np.asarray(psfim, dtype='f4')

    shape = psfim.shape
    assert shape[0] == shape[1], 'require square psf images'
    assert shape[0] % 2 != 0, 'require odd psf images'

    return psfim


def _extract_psf_image_fix(exposure, orig_cen):
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

    psfim = np.asarray(psfim, dtype='f4')

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
        weight[:, :] = 1.0 / medvar
    else:
        print('    weight is all zero, found none that passed cuts')

    return weight


def _extract_jacobian_at_source(exp, source):
    """
    extract an ngmix.Jacobian from the image object
    and object record

    exp: lsst.afw.image.ExposureF
        An exposure object
    source: lsst.afw.table.SourceRecord
        The source record created during detection

    returns
    --------
    Jacobian: ngmix.Jacobian
        The local jacobian
    """

    orig_cen = exp.getWcs().skyToPixel(source.getCoord())

    if np.isnan(orig_cen.getY()):
        LOG.info('falling back on integer location')
        # fall back to integer pixel location
        peak = source.getFootprint().getPeaks()[0]
        orig_cen_i = peak.getI()
        orig_cen = geom.Point2D(
            x=orig_cen_i.getX(),
            y=orig_cen_i.getY(),
        )

    return get_jacobian(exp, orig_cen)


def get_output_dtype():
    dt = [
        ('stamp_size', 'i4'),
        ('row0', 'i4'),  # bbox row start
        ('col0', 'i4'),  # bbox col start
        ('row', 'f4'),  # row in image. Use row0 to get to global pixel coords
        ('col', 'f4'),  # col in image. Use col0 to get to global pixel coords
        ('row_diff', 'f4'),  # difference from peak location
        ('col_diff', 'f4'),  # difference from peak location
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('psfrec_flags', 'i4'),  # psfrec is the original psf
        ('psfrec_g', 'f8', 2),
        ('psfrec_T', 'f8'),
        # values from .mask of input exposures
        ('bmask', 'i4'),
        # values for ormask across all input exposures to coadd
        ('ormask', 'i4'),
        # fraction of images going into a pixel that were masked
        ('mfrac', 'f4'),
    ]

    return dt


def get_output_struct(res):
    """
    get the output struct

    Parameters
    ----------
    res: ndarray
        The result from running metadetect.fitting.fit_mbobs_wavg

    Returns
    -------
    ndarray
        Has the fields from res, with new fields added, see get_output_dtype
    """
    dt = get_output_dtype()
    output = eu.numpy_util.add_fields(res, dt)

    for subdt in dt:
        name = subdt[0]
        dtype = subdt[1]

        if 'flags' in name:
            output[name] = NO_ATTEMPT
        elif name in ('bmask', 'ormask'):
            output[name] = 0
        elif dtype[0] == 'i':
            output[name] = -9999
        else:
            output[name] = np.nan

    return output


def get_output(wcs, source, res, bmask, stamp_size, exp_bbox):
    """
    get the output structure, copying in results

    The following fields are not set:
        psfrec_flags, psfrec_g, psfrec_T
        mfrac

    Parameters
    ----------
    wcs: a stack wcs
        The wcs with which to determine the ra, dec
    res: ndarray
        The result from running metadetect.fitting.fit_mbobs_wavg
    bmask: int
        The bmask value at the location of this object
    stamp_size: int
        The stamp size used for the measurement
    exp_bbox: lsst.geom.Box2I
        The bounding box used for measurement

    Returns
    -------
    ndarray
        Has the fields from res, with new fields added, see get_output_dtype
    """
    output = get_output_struct(res)

    orig_cen = source.getCentroid()

    skypos = wcs.pixelToSky(orig_cen)

    peak = source.getFootprint().getPeaks()[0]
    peak_loc = peak.getI()

    if np.isnan(orig_cen.getY()):
        orig_cen = peak.getCentroid()
        cen_offset = geom.Point2D(np.nan, np.nan)
    else:
        cen_offset = geom.Point2D(
            orig_cen.getX() - peak_loc.getX(),
            orig_cen.getY() - peak_loc.getY(),
        )

    output['stamp_size'] = stamp_size
    output['row0'] = exp_bbox.getBeginY()
    output['col0'] = exp_bbox.getBeginX()
    output['row'] = orig_cen.getY()
    output['col'] = orig_cen.getX()
    output['row_diff'] = cen_offset.getY()
    output['col_diff'] = cen_offset.getX()

    output['ra'] = skypos.getRa().asDegrees()
    output['dec'] = skypos.getDec().asDegrees()

    # remove DETECTED bit, it is just clutter since all detected
    # objects have this bit set
    detected = afw_image.Mask.getPlaneBitMask('DETECTED')
    output['bmask'] = bmask & ~detected

    return output


class MissingDataError(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


class AllZeroWeightError(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)
