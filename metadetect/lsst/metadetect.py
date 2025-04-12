import logging
import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure
from lsst.pex.config import (
    Config,
    ChoiceField,
    ConfigField,
    ConfigurableField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import Task
from lsst.meas.algorithms import SourceDetectionTask

from .. import procflags

from .skysub import subtract_sky_mbexp

from .defaults import (
    DEFAULT_FWHM_REG,
    DEFAULT_FWHM_SMOOTH,
    DEFAULT_STAMP_SIZES,
    DEFAULT_SUBTRACT_SKY,
    DEFAULT_WEIGHT_FWHMS,
)
from . import measure
from .metacal_exposures import get_metacal_mbexps_fixnoise
from .util import get_integer_center, get_jacobian, override_config

LOG = logging.getLogger('lsst_metadetect')


def run_metadetect(
    mbexp, noise_mbexp, rng, mfrac_mbexp=None, ormasks=None, config=None, show=False,
):
    """
    Run metadetection on the input MultiBandObsList

    Note that bright object masking must be applied outside of this function.

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    noise_mbexp: lsst.afw.image.MultibandExposure
        The noise exposures for metacal
    mfrac_mbexp: lsst.afw.image.MultibandExposure, optional
        The fraction of masked exposures for the pixel; for coadds this is the
        fraction of input images contributing to each pixel that were masked
    ormasks: list of images, optional
        A list of logical or masks, such as created for all images that went
        into a coadd.

        Note when coadding an ormask is created in the .mask attribute. But
        this code expects the mask attribute for each exposure to be not an or
        of all masks from the original exposures, but a mask indicating problem
        areas such as bright objects or apodized edges.

        In the future we expect the MultibandExposure to have an ormask attribute
    rng: np.random.RandomState
        Random number generator
    config: dict, optional
        Configuration for the fitter, metacal, psf, detect, Entries
        in this dict override defaults; see lsst_configs.py
    show: bool, optional
        if set to True images will be shown

    Returns
    -------
    result dict
        This is keyed by shear string 'noshear', '1p', ... or None if there was
        a problem doing the metacal steps; this only happens if the setting
        metacal_psf is set to 'fitgauss' and the fitting fails
    """

    config_override = config if config is not None else {}
    config = MetadetectConfig()
    config.setDefaults()

    override_config(config, config_override)

    config.freeze()
    config.validate()
    task = MetadetectTask(config=config)
    result = task.run(mbexp, noise_mbexp, rng, mfrac_mbexp, ormasks, show=show,)
    return result


class WeightConfig(Config):
    fwhm = Field[float](
        doc="FWHM of the Gaussian weight function (in arcseconds)",
        default=None,
    )
    fwhm_smooth = Field[float](
        doc="FWHM of the Gaussian smoothing function (in arcseconds)",
        default=DEFAULT_FWHM_SMOOTH,
    )
    fwhm_reg = Field[float](
        doc="FWHM of the Gaussian regularization function (in arcseconds)",
        default=DEFAULT_FWHM_REG,
    )


class LmParseConfig(Config):
    maxfev = Field[int](
        doc="Maximum number of function evaluations",
        default=4000,
    )
    ftol = Field[float](
        doc="Relative error desired in the sum of squares",
        default=1.0e-5,
    )
    xtol = Field[float](
        doc="Relative error desired in the approximate solution",
        default=1.0e-5,
    )


class PsfConfig(Config):
    """Config class for fitting original PSFs."""
    model = ChoiceField[str](
        doc="Model for fitting original PSFs",
        default="am",
        allowed={
            "am": "Adaptive moments",  # TODO: What are the other possible values?
            "gauss": "Gaussian",
        },
        optional=False,
    )
    ntry = Field[int](
        doc="Number of tries for fitting original PSFs",
        default=4,
    )
    lm_pars = ConfigField[LmParseConfig](
        doc="Config for Levenberg-Marquardt fitting",
    )


class MetacalConfig(Config):
    use_noise_image = Field[bool](
        doc="Whether to use the noise image",
        default=True,
    )
    psf = ChoiceField[str](
        doc="PSF model",
        default="fitgauss",
        allowed={
            "fitgauss": "Fit a Gaussian to the PSF",
        }
    )
    types = ListField[str](
        doc="List of artificial shears to apply.",
        default=["noshear", "1p", "1m",],
    )

    def validate(self):
        super().validate()
        if not set(self.types).issubset({"noshear", "1p", "1m", "2p", "2m"}):
            raise FieldValidationError(
                self.__class__.types,
                self,
                "types must be a list consisting of any combinations of "
                "{'noshear', '1p', '1m', '2p', '2m'}",
            )


class MetadetectConfig(Config):
    subtract_sky = Field[bool](
        doc="Whether to subtract the sky before running metadetect",
        default=DEFAULT_SUBTRACT_SKY,
    )
    meas_type = ChoiceField[str](
        doc="Measurement type",
        default="wmom",
        allowed={
            "wmom": "Weighted moments",
            "ksigma": "Fourier moments",  # TODO: improve docstring
            "pgauss": "Pre-seeing moments",
            "am": "Adaptive moments",
        }
    )

    weight = ConfigField[WeightConfig](
        doc="FWHM of the Gaussian weight function (in arcseconds)",
    )

    psf = ConfigField[PsfConfig](
        doc="Config for fitting the original PSFs",
    )

    detect = ConfigurableField(
        doc="Detection config",
        target=SourceDetectionTask,
    )
    metacal = ConfigField[MetacalConfig](
        doc="Config for metacal",
    )

    @property
    def stamp_size(self):
        # Caution: This provides a backdoor entry to change the behavior
        # by changing the DEFAULT_STAMP_SIZES.
        return DEFAULT_STAMP_SIZES[self.meas_type]

    def setDefaults(self):
        super().setDefaults()
        self.weight.fwhm = DEFAULT_WEIGHT_FWHMS.get(self.meas_type, None)

    def validate(self):
        super().validate()

        if self.stamp_size != DEFAULT_STAMP_SIZES[self.meas_type]:
            raise FieldValidationError(
                self.__class__.stamp_size,
                self,
                f"stamp_size must be {DEFAULT_STAMP_SIZES[self.meas_type]} "
                f"for meas_type {self.meas_type}",
            )


class MetadetectTask(Task):
    ConfigClass = MetadetectConfig
    _DefaultName = "metadetect"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("detect")

    def run(
        self,
        mbexp,
        noise_mbexp,
        rng,
        mfrac_mbexp=None,
        ormasks=None,
        show=False,
    ):

        # This is to support methods that are not yet refactored.
        config = self.config.toDict()
        # Because this is a property and not a Field, we set this explicitly.
        config['stamp_size'] = self.config.stamp_size
        config['detect']['thresh'] = self.detect.config.thresholdValue

        ormask = combine_ormasks(mbexp, ormasks)
        mfrac, wgts = get_mfrac_mbexp(mbexp=mbexp, mfrac_mbexp=mfrac_mbexp)

        if self.config.subtract_sky:
            subtract_sky_mbexp(mbexp=mbexp, thresh=self.config.detect.thresholdValue)

        psf_stats = fit_original_psfs_mbexp(
            mbexp=mbexp,
            wgts=wgts,
            rng=rng,
        )

        fitter = get_fitter(config, rng=rng)

        metacal_types = config['metacal'].get('types', None)

        mdict, noise_mdict = get_metacal_mbexps_fixnoise(
            mbexp=mbexp, noise_mbexp=noise_mbexp, types=metacal_types,
        )

        result = {}
        for shear_str, mcal_mbexp in mdict.items():

            res = detect_deblend_and_measure(
                mbexp=mcal_mbexp,
                fitter=fitter,
                config=config,
                rng=rng,
                show=show,
            )

            if res is not None:
                band = mcal_mbexp.bands[0]
                exp = mcal_mbexp[band]

                add_mfrac(config=config, mfrac=mfrac, res=res, exp=exp)
                add_ormask(ormask, res)
                add_original_psf(psf_stats, res)

            result[shear_str] = res

        return result


def detect_deblend_and_measure(
    mbexp,
    fitter,
    config,
    rng,
    show=False,
):
    """
    run detection, deblending and measurements.

    Note deblending is always run in a hierarchical detection process, but the
    deblending is only used for getting centers, and because there is currently
    no other way to split footprints

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The metacal'ed exposures to process
    fitter: e.g. ngmix.gaussmom.GaussMom or ngmix.ksigmamom.KSigmaMom
        For calculating moments
    config: dict, optional
        Configuration for the fitter, metacal, psf, detect, Entries
        in this dict override defaults; see lsst_configs.py
    thresh: float
        The detection threshold in units of the sky noise
    stamp_size: int
        Size for postage stamps.
    show: bool, optional
        If set to True, show images during processing
    """

    LOG.info('measuring with blended stamps')

    sources, detexp = measure.detect_and_deblend(
        mbexp=mbexp,
        rng=rng,
        thresh=config['detect']['thresh'],
        show=show,
    )

    if config['weight'] is not None:
        fwhm_reg = config['weight'].get('fwhm_reg', 0)
    else:
        fwhm_reg = 0

    results = measure.measure(
        mbexp=mbexp,
        detexp=detexp,
        sources=sources,
        fitter=fitter,
        stamp_size=config['stamp_size'],
        fwhm_reg=fwhm_reg,
    )

    return results


def add_mfrac(config, mfrac, res, exp):
    """
    calculate and add mfrac to the input result array
    """
    if np.any(mfrac > 0):

        # we are using the positions with the metacal shear removed for
        # this.

        cen, _ = get_integer_center(
            wcs=exp.getWcs(),
            bbox=exp.getBBox(),
            as_double=True,
        )
        jac = get_jacobian(exp=exp, cen=cen)

        res['mfrac'] = measure_weighted_mfrac(
            mfrac=mfrac,
            x=res['col'] - res['col0'],
            y=res['row'] - res['row0'],
            jac=jac,
            fwhm=config.get('mfrac_fwhm', None),
        )
    else:
        res['mfrac'] = 0


def measure_weighted_mfrac(
    *,
    mfrac,
    x,
    y,
    jac,
    fwhm,
):
    """
    Measure a Gaussian-weighted average of an image.

    This function is meant to be used with images that represent the fraction
    of single-epoch images that are masked in each pixel of a coadd. It
    computes a Gaussian-weighted average of the image at a list of locations.

    Parameters
    ----------
    mfrac : np.ndarray
        The input image with which to compute the weighted averages.
    x : np.ndarray
        The input x/col values for the positions at which to compute the
        weighted average.
    y : np.ndarray
        The input y/row values for the positions at which to compute the
        weighted average.
    box_sizes : np.ndarray
        The size of the stamp to use to measure the weighted average. Should be
        big enough to hold 2 * `fwhm`.
    jac: ngmix.Jacobian
        The jacobian for the data
    fwhm : float or None
        The FWHM of the Gaussian aperture in arcseconds. If None, a default
        of 1.2 is used.

    Returns
    -------
    mfracs : np.ndarray
        The weighted averages at each input location.
    """

    if fwhm is None:
        fwhm = 1.2

    ny, nx = mfrac.shape

    gauss_wgt = ngmix.GMixModel(
        [0, 0, 0, 0, ngmix.moments.fwhm_to_T(fwhm), 1],
        'gauss',
    )
    sigma = ngmix.moments.fwhm_to_sigma(fwhm)
    box_rad = int(round(sigma * 5))

    mfracs = []
    for i in range(x.shape[0]):
        ix = int(np.floor(x[i] + 0.5))
        iy = int(np.floor(y[i] + 0.5))

        xstart = ix - box_rad
        xend = ix + box_rad + 1
        ystart = iy - box_rad
        yend = iy + box_rad + 1

        if xstart < 0:
            xstart = 0
        if ystart < 0:
            ystart = 0
        if xend > nx:
            xend = nx
        if yend > ny:
            yend = ny

        sub_mfrac = mfrac[xstart:xend, ystart:yend]
        if sub_mfrac.size == 0:
            mfracs.append(1.0)
        else:
            cy, cx = (y[i] - ystart, x[i] - xstart)
            this_jac = jac.copy()
            this_jac.set_cen(row=cy, col=cx)

            obs = ngmix.Observation(
                image=sub_mfrac,
                jacobian=this_jac,
            )

            stats = gauss_wgt.get_weighted_sums(obs, maxrad=box_rad)

            # this is the weighted average in the image using the
            # Gaussian as the weight.
            mfracs.append(stats["sums"][5] / stats["wsum"])

    return np.array(mfracs)


def add_ormask(ormask, res):
    """
    copy in ormask values using the row, col positions
    """
    for i in range(res.size):
        row_diff = res['row'][i] - res['row0'][i]
        col_diff = res['col'][i] - res['col0'][i]
        local_row = int(np.floor(row_diff + 0.5))
        local_col = int(np.floor(col_diff + 0.5))

        res['ormask'][i] = ormask[local_row, local_col]


def add_original_psf(psf_stats, res):
    """
    copy in psf results
    """
    res['psfrec_flags'][:] = psf_stats['flags']
    res['psfrec_g'][:, 0] = psf_stats['g1']
    res['psfrec_g'][:, 1] = psf_stats['g2']
    res['psfrec_T'][:] = psf_stats['T']


def get_fitter(config, rng=None):
    """
    get the fitter based on the 'fitter' input
    """

    meas_type = config['meas_type']

    if meas_type == 'am':
        fitter_obj = ngmix.admom.AdmomFitter(rng=rng)
        guesser = ngmix.guessers.GMixPSFGuesser(
            rng=rng, ngauss=1, guess_from_moms=True,
        )
        fitter = ngmix.runners.Runner(
            fitter=fitter_obj, guesser=guesser,
            ntry=2,
        )
        # TODO is there a better way?
        fitter.kind = 'am'

    elif meas_type == 'em':
        fitter = None
    else:
        fwhm = config['weight']['fwhm']

        if meas_type == 'wmom':
            fitter = ngmix.gaussmom.GaussMom(fwhm=fwhm)
        else:
            # for backwards compatibility with older ngmix that did not
            # support the fwhm_smooth keyword.
            sm_kwargs = {}
            if config['weight']['fwhm_smooth'] > 0:
                sm_kwargs["fwhm_smooth"] = config['weight']['fwhm_smooth']

            if meas_type == 'ksigma':
                fitter = ngmix.ksigmamom.KSigmaMom(
                    fwhm=fwhm,
                    **sm_kwargs
                )
            elif meas_type == 'pgauss':
                fitter = ngmix.prepsfmom.PGaussMom(
                    fwhm=fwhm,
                    **sm_kwargs
                )
            else:
                raise ValueError("bad meas_type: '%s'" % meas_type)

    return fitter


def combine_ormasks(mbexp, ormasks):
    """
    logical or together all the ormasks, or if ormasks is None create zeroed
    versions for each band
    """
    if ormasks is None:
        bands = mbexp.bands
        dims = mbexp[bands[0]].image.array.shape
        ormask = np.zeros(dims, dtype='i4')
    else:
        for imask, tormask in enumerate(ormasks):
            if imask == 0:
                ormask = tormask.copy()
            else:
                ormask |= tormask

    return ormask


def get_mfrac_mbexp(mbexp, mfrac_mbexp):
    """
    set the masked fraction image, averaged over all bands

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process

    Returns
    -------
    mfrac: array
    """
    wgts = []

    wsum = 0.0

    mfrac = None
    for exp, mfrac_exp in zip(mbexp, mfrac_mbexp):
        varray = exp.variance.array
        w = np.where(np.isfinite(varray) & (varray > 0))

        if w[0].size == 0:
            raise ValueError('no variance are finite')

        var = np.median(exp.variance.array[w])
        wgt = 1/var
        wgts.append(wgt)
        wsum += wgt

        if mfrac is None:
            mfrac = wgt * mfrac_exp.image.array
        else:
            mfrac += wgt * mfrac_exp.image.array

    mfrac *= 1.0/wsum

    return mfrac, wgts


def fit_original_psfs_mbexp(mbexp, rng, wgts):
    """
    fit the original psfs at the center of the image and get the mean g1,g2,T
    across all bands

    This can fail and flags will be set, but we proceed
    """

    assert len(wgts) == len(mbexp)
    wsum = sum(wgts)
    if wsum <= 0:
        raise ValueError(f'got sum(wgts) = {wsum}')

    fitter = ngmix.admom.AdmomFitter(rng=rng)
    guesser = ngmix.guessers.GMixPSFGuesser(
        rng=rng, ngauss=1, guess_from_moms=True,
    )
    runner = ngmix.runners.PSFRunner(fitter=fitter, guesser=guesser, ntry=4)

    try:
        g1sum = 0.0
        g2sum = 0.0
        Tsum = 0.0

        for exp, wgt in zip(mbexp, wgts):
            cen, _ = get_integer_center(
                wcs=exp.getWcs(),
                bbox=exp.getBBox(),
                as_double=True,
            )
            jac = get_jacobian(exp=exp, cen=cen)

            psf_im = measure.extract_psf_image(exp, cen)

            psf_cen = (np.array(psf_im.shape)-1.0)/2.0
            psf_jacob = jac.copy()
            psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

            psf_obs = ngmix.Observation(
                psf_im,
                jacobian=psf_jacob,
            )
            res = runner.go(obs=psf_obs)
            if res['flags'] != 0:
                raise BootPSFFailure('failed to fit psf')

            g1, g2 = res['e']
            T = res['T']

            g1sum += g1*wgt
            g2sum += g2*wgt
            Tsum += T*wgt

        g1 = g1sum/wsum
        g2 = g2sum/wsum
        T = Tsum/wsum

        flags = 0

    except BootPSFFailure:
        flags = procflags.PSF_FAILURE
        g1 = -9999.0
        g2 = -9999.0
        T = -9999.0

    return {
        'flags': flags,
        'g1': g1,
        'g2': g2,
        'T': T,
    }
