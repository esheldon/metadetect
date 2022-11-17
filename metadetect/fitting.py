import logging
import copy

import numpy as np

import ngmix
from ngmix.gexceptions import (
    BootPSFFailure, PSFFluxFailure,
)
from ngmix.moments import make_mom_result, fwhm_to_T
from pkg_resources import parse_version
from ngmix.bootstrap import bootstrap
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import SimplePSFGuesser
from ngmix.fitting import Fitter

from .util import Namer
from . import procflags

MAX_NUM_SHEAR_BANDS = 6

logger = logging.getLogger(__name__)


if parse_version(ngmix.__version__) < parse_version("2.1.0"):
    MOMNAME = "mom"
else:
    MOMNAME = "sums"


def fit_mbobs_gauss(
    *,
    mbobs,
    bmask_flags,
    rng,
    shear_bands=None,
):
    """Fit a multiband obs using a Gaussian fit.

    Parameters
    ----------
    mbobs : ngmix.MultiBandObsList
        The observation to use for shear measurement.
    bmask_flags : int
        Observations with these bits set in the bmask are not fit.
    rng : np.random.RandomState
        Random state for fitting.
    shear_bands : list of int, optional
        A list of indices into each mbobs that denotes which band is used for shear.
        Default is to use all bands.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    if shear_bands is None:
        shear_bands = list(range(len(mbobs)))

    res = get_wavg_output_struct(len(mbobs), "gauss", shear_bands=shear_bands)

    flags = 0
    for obslist in mbobs:
        if len(obslist) == 0:
            flags |= procflags.MISSING_BAND
            continue

        if len(obslist) > 1:
            flags |= procflags.INCONSISTENT_BANDS
            continue

        for obs in obslist:
            if not np.any(obs.weight > 0):
                flags |= procflags.ZERO_WEIGHTS

            if np.any((obs.bmask & bmask_flags) != 0):
                flags |= procflags.EDGE_HIT

    if any(s >= len(mbobs) for s in shear_bands):
        flags |= procflags.INCONSISTENT_BANDS

    if flags == 0:
        shear_mbobs = ngmix.MultiBandObsList()
        for band in shear_bands:
            shear_mbobs.append(mbobs[band])

        try:
            ores = bootstrap(
                shear_mbobs,
                _make_obj_runner(rng, shear_mbobs),
                psf_runner=_make_psf_runner(rng),
                ignore_failed_psf=False,
            )
        except BootPSFFailure:
            flags |= procflags.PSF_FAILURE
        except PSFFluxFailure:
            flags |= procflags.PSF_FAILURE
        except Exception:
            flags |= procflags.OBJ_FAILURE
            flags |= procflags.PSF_FAILURE

    if flags == 0:
        res["gauss_obj_flags"] = ores["flags"]
        res["gauss_T_flags"] = ores["flags"]
        res["gauss_s2n"] = ores["s2n"]
        res["gauss_g"] = ores["g"]
        res["gauss_g_cov"] = ores["g_cov"]
        res["gauss_T"] = ores["T"]
        res["gauss_T_err"] = ores["T_err"]

        pflags = 0
        psf_g_sum = np.zeros(2)
        psf_T_sum = 0.0
        wgt_sum = 0.0
        for obslist in shear_mbobs:
            for obs in obslist:
                pflags |= obs.psf.meta["result"]["flags"]
                if obs.psf.meta["result"]["flags"] == 0:
                    msk = obs.weight > 0
                    _wgt = np.median(obs.weight[msk])
                    psf_T_sum += obs.psf.meta["result"]["T"] * _wgt
                    psf_g_sum += (
                        obs.psf.meta["result"]["g"]
                        * _wgt
                        * obs.psf.meta["result"]["T"]
                    )
                    wgt_sum += _wgt

        res["gauss_psf_flags"] = pflags
        if res["gauss_psf_flags"] == 0:
            res["gauss_psf_T"] = psf_T_sum / wgt_sum
            res["gauss_psf_g"] = psf_g_sum / psf_T_sum
            res["gauss_T_ratio"] = res["gauss_T"] / res["gauss_psf_T"]

        res["gauss_flags"] = res["gauss_obj_flags"] | res["gauss_psf_flags"]
    else:
        res["gauss_flags"] |= flags

    return res


def _make_psf_runner(rng):
    psf_guesser = SimplePSFGuesser(
        rng=rng,
        guess_from_moms=True,
    )
    psf_fitter = Fitter(model="gauss")
    psf_runner = PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    return psf_runner


def _make_obj_runner(rng, mbobs):
    obs = mbobs[0][0]
    nband = len(mbobs)
    scale = obs.jacobian.get_scale()
    prior = _make_ml_prior(rng, scale, nband)

    fitter = ngmix.fitting.Fitter(model='gauss', prior=prior)
    guesser = ngmix.guessers.TPSFFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )
    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    return runner


def _make_ml_prior(rng, scale, nband):
    """make the prior for the fitter.

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    nband: int
        number of bands
    """
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )
    T_prior = ngmix.priors.TwoSidedErf(
        minval=-10.0,
        width_at_min=0.03,
        maxval=1.0e6,
        width_at_max=1.0e5,
        rng=rng,
    )
    F_prior = ngmix.priors.TwoSidedErf(
        minval=-1.0e4,
        width_at_min=1.0,
        maxval=1.0e9,
        width_at_max=0.25e8,
        rng=rng,
    )
    F_prior = [F_prior] * nband

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior


def fit_mbobs_list_joint(
    *, mbobs_list, fitter_name, bmask_flags, rng, shear_bands=None,
):
    """Fit the ojects in a list of ngmix.MultiBandObsList using a joint fitter.

    The fitter is run per object on the bands used for shear.

    Parameters
    ----------
    mbobs_list : a list of ngmix.MultiBandObsList
        The observations to use for shear measurement.
    fitter_name : str
        The name of the fitter to use.
    bmask_flags : int
        Observations with these bits set in the bmask are not fit.
    shear_bands : list of int, optional
        A list of indices into each mbobs that denotes which band is used for shear.
        Default is to use all bands.
    rng : np.random.RandomState
        Random state for fitting.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    if fitter_name in ["am", "admom"]:
        fit_func = fit_mbobs_admom
    elif fitter_name == "gauss":
        fit_func = fit_mbobs_gauss
    else:
        raise RuntimeError("Joint fitter '%s' not recognized!" % fitter_name)

    res = []
    for i, mbobs in enumerate(mbobs_list):
        _res = fit_func(
            mbobs=mbobs,
            bmask_flags=bmask_flags,
            shear_bands=shear_bands,
            rng=rng,
        )
        res.append(_res)

    if len(res) > 0:
        return np.hstack(res)
    else:
        return None


def get_admom_fitter(rng):
    fitter = ngmix.admom.AdmomFitter(rng=rng)
    guesser = ngmix.guessers.GMixPSFGuesser(
        rng=rng, ngauss=1, guess_from_moms=True,
    )
    return ngmix.runners.Runner(
        fitter=fitter, guesser=guesser,
        ntry=2,
    )


def fit_mbobs_admom(
    *,
    mbobs,
    bmask_flags,
    rng,
    shear_bands=None,
):
    """Fit a multiband obs using adaptive moments.

    This function forms a coadd of the shear bands and then runs
    adaptive moments on the coadd.

    Parameters
    ----------
    mbobs : ngmix.MultiBandObsList
        The observation to use for shear measurement.
    bmask_flags : int
        Observations with these bits set in the bmask are not fit.
    rng : np.random.RandomState
        Random state for fitting.
    shear_bands : list of int, optional
        A list of indices into each mbobs that denotes which band is used for shear.
        Default is to use all bands.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    fitter = get_admom_fitter(rng)
    nband = len(mbobs)
    if shear_bands is None:
        shear_bands = list(range(len(mbobs)))
    res = get_wavg_output_struct(nband, "am", shear_bands=shear_bands)

    flags = 0
    for obslist in mbobs:
        if len(obslist) == 0:
            flags |= procflags.MISSING_BAND
            continue

        if len(obslist) > 1:
            flags |= procflags.INCONSISTENT_BANDS
            continue

        for obs in obslist:
            if not np.any(obs.weight > 0):
                flags |= procflags.ZERO_WEIGHTS

            if np.any((obs.bmask & bmask_flags) != 0):
                flags |= procflags.EDGE_HIT

    if flags == 0:
        # first we coadd the shear bands
        coadd_obs, coadd_flags = make_coadd_obs(mbobs, shear_bands=shear_bands)
        flags |= coadd_flags

    if flags == 0:
        # then fit the PSF
        pres = fitter.go(coadd_obs.psf)
        res["am_psf_flags"] = pres["flags"]
        if pres["flags"] == 0:
            res["am_psf_g"] = pres["e"]
            res["am_psf_T"] = pres["T"]

        # then fit the object
        sym_coadd_obs = symmetrize_obs_weights(coadd_obs)
        gres = fitter.go(sym_coadd_obs)
        res["am_T_flags"] = gres["T_flags"]
        if gres["T_flags"] == 0:
            res["am_T"] = gres["T"]
            res["am_T_err"] = gres["T_err"]
            if pres["flags"] == 0:
                res["am_T_ratio"] = res["am_T"] / res["am_psf_T"]

        res["am_obj_flags"] = gres["flags"]
        if gres["flags"] == 0:
            res["am_s2n"] = gres["s2n"]
            res["am_g"] = gres["e"]
            res["am_g_cov"] = gres["e_cov"]

        # this replaces the flags so they are zero and unsets the default of
        # no attempt
        res["am_flags"] = (res["am_psf_flags"] | res["am_obj_flags"])
    else:
        # this branch ensures noattempt remains set
        res["am_flags"] |= flags

    return res


def make_coadd_obs(mbobs, shear_bands=None):
    """Coadd the observations in an ngmix mbobs assuming they all have the same
    shaped images and same Jacobians.

    Parameters
    ----------
    mbobs : ngmix.MultibandObsList
        The observation to use for shear measurement.
    shear_bands : list of int, optional
        A list of indices into each mbobs that denotes which band is used for shear.
        Default is to use all bands.

    Returns
    -------
    coadd_obs : ngmix.Observation
        The coadded observation with the PSF set.
    flags : int
        Any flags for errors in making the coadd obs.
    """
    nbands = len(mbobs)
    flags = 0
    if shear_bands is None:
        shear_bands = list(range(nbands))

    # if we have one input band, then simply
    # return if if the data is consistent
    if nbands == 1:
        if shear_bands[0] == 0:
            return mbobs[0][0], flags
        else:
            flags |= procflags.INCONSISTENT_BANDS
            return None, flags

    # if we ask for a single band for shear, then return that band
    # if the data is consistent
    if len(shear_bands) == 1:
        if shear_bands[0] < nbands:
            return mbobs[shear_bands[0]][0], flags
        else:
            flags |= procflags.INCONSISTENT_BANDS
            return None, flags

    # at this point, we are coadding more than one band
    # let's check for consistent data

    # we can't use shear bands we do not have
    if any(s >= nbands for s in shear_bands):
        flags |= procflags.INCONSISTENT_BANDS
        return None, flags

    # all of the image shapes and jacobians have to be the same
    fobs = mbobs[shear_bands[0]][0]
    for shear_band in shear_bands[1:]:
        obs = mbobs[shear_band][0]
        if (
            (repr(fobs.jacobian) != repr(obs.jacobian))
            or (fobs.image.shape != obs.image.shape)
            or (not fobs.has_psf())
            or (not obs.has_psf())
            or (repr(fobs.psf.jacobian) != repr(obs.psf.jacobian))
            or (fobs.psf.image.shape != obs.psf.image.shape)
        ):
            flags |= procflags.INCONSISTENT_BANDS
            return None, flags

    # if we get here, we need to coadd
    # first make the weights
    wgts = np.zeros(len(shear_bands))
    for i, band in enumerate(shear_bands):
        obs = mbobs[band][0]
        msk = obs.weight > 0
        if not np.any(msk):
            flags |= procflags.ZERO_WEIGHTS
            # let's not waste time since we are missing a band
            return None, flags
        else:
            wgts[i] = np.median(obs.weight[msk])
    wgts /= np.sum(wgts)

    # finish coadding
    image = np.zeros_like(mbobs[shear_bands[0]][0].image)
    weight = np.zeros_like(mbobs[shear_bands[0]][0].weight)
    mfrac = np.zeros_like(mbobs[shear_bands[0]][0].image)
    nmfrac = 0
    noise = np.zeros_like(mbobs[shear_bands[0]][0].image)
    nnoise = 0
    bmask = np.zeros_like(mbobs[shear_bands[0]][0].image, dtype=np.int32)
    nbmask = 0
    ormask = np.zeros_like(mbobs[shear_bands[0]][0].image, dtype=np.int32)
    normask = 0
    meta = {}
    psf_meta = {}
    psf_image = np.zeros_like(mbobs[shear_bands[0]][0].psf.image)

    for i, shear_band in enumerate(shear_bands):
        obs = mbobs[shear_band][0]
        wgt = wgts[i]

        meta.update(obs.meta)
        meta.update(mbobs[shear_band].meta)
        psf_meta.update(obs.psf.meta)

        psf_image += (wgt * obs.psf.image)

        image += (wgt * obs.image)
        weight += (wgt**2 / obs.weight)

        if obs.has_mfrac():
            mfrac += (wgt * obs.mfrac)
            nmfrac += 1

        if obs.has_noise():
            noise += (wgt * obs.noise)
            nnoise += 1

        if obs.has_bmask():
            bmask |= obs.bmask
            nbmask += 1

        if obs.has_ormask():
            ormask |= obs.ormask
            normask += 1

    weight = 1.0 / weight
    msk = ~np.isfinite(weight)
    if np.any(msk):
        weight[msk] = 0

    if np.all(weight == 0):
        flags |= procflags.ZERO_WEIGHTS
        return None, flags

    for var in [nmfrac, nnoise, nbmask, normask]:
        if var not in [0, len(shear_bands)]:
            flags |= procflags.INCONSISTENT_BANDS
            return None, flags

    meta.update(mbobs.meta)

    kwargs = {
        "weight": weight,
        "jacobian": mbobs[shear_bands[0]][0].jacobian,
        "psf": ngmix.Observation(
            psf_image,
            jacobian=mbobs[shear_bands[0]][0].psf.jacobian,
            meta=psf_meta,
        ),
        "meta": meta,
    }
    if nmfrac > 0:
        kwargs["mfrac"] = mfrac
    if nnoise > 0:
        kwargs["noise"] = noise
    if nbmask > 0:
        kwargs["bmask"] = bmask
    if normask > 0:
        kwargs["ormask"] = ormask

    cobs = ngmix.Observation(image, **kwargs)

    return cobs, flags


def get_coellip_ngauss(name):
    ngauss = int(name[7:])
    return ngauss


def fit_all_psfs(mbobs, rng):
    """
    measure all psfs in the input observations and store the results
    in the meta dictionary, and possibly as a gmix for model fits

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to fit
    psf_conf: dict
        Config for the measurements/fitting
    rng: np.random.RandomState
        The random number generator, used for guessers
    """
    fitter = ngmix.admom.AdmomFitter(rng=rng)
    guesser = ngmix.guessers.GMixPSFGuesser(
        rng=rng, ngauss=1, guess_from_moms=True,
    )

    runner = ngmix.runners.PSFRunner(
        fitter=fitter, guesser=guesser, ntry=10,
    )

    for obslist in mbobs:
        assert len(obslist) == 1, 'metadetect is not multi-epoch'

        obs = obslist[0]
        runner.go(obs=obs)

        flags = obs.psf.meta['result']['flags']
        if flags != 0:
            raise BootPSFFailure("failed to measure psfs: %s" % flags)


def fit_mbobs_list_wavg(
    *, mbobs_list, fitter, bmask_flags, shear_bands=None, fwhm_reg=0
):
    """Fit the ojects in a list of ngmix.MultiBandObsList using a weighted average
    over bands.

    The fitter is run per-band and then results are combined across bands
    via a weighted average.

    Parameters
    ----------
    mbobs_list : a list of ngmix.MultiBandObsList
        The observations to use for shear measurement.
    fitter : ngmix fitter
        The fitter to use per band per MultiBandObsList.
    bmask_flags : int
        Observations with these bits set in the bmask are not fit.
    shear_bands : list of int, optional
        A list of indices into each mbobs that denotes which band is used for shear.
        Default is to use all bands.
    fwhm_reg : float, optional
        This value is converted to T and used to regularize the shapes via

            e_{1,2} = M_{1,2}/(T + T_reg)

        For Gaussians, this relationship is equivalent to smoothing by a round
        Gaussian with FWHM `fwhm_reg`.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    res = []
    for i, mbobs in enumerate(mbobs_list):

        _res = fit_mbobs_wavg(
            mbobs=mbobs,
            fitter=fitter,
            bmask_flags=bmask_flags,
            shear_bands=shear_bands,
            fwhm_reg=fwhm_reg,
        )
        res.append(_res)

    if len(res) > 0:
        return np.hstack(res)
    else:
        return None


def fit_mbobs_wavg(
    *,
    mbobs,
    fitter,
    bmask_flags,
    shear_bands=None,
    fwhm_reg=0,
):
    """Fit the object in the ngmix.MultiBandObsList using a weighted average
    over bands.

    The fitter is run per-band and then results are combined across bands
    via a weighted average.

    Parameters
    ----------
    mbobs : ngmix.MultiBandObsList
        The observation to use for shear measurement.
    fitter : ngmix fitter
        The fitter to use per band.
    bmask_flags : int
        Observations with these bits set in the bmask are not fit.
    shear_bands : list of int, optional
        A list of indices into each mbobs that denotes which band is used for shear.
        Default is to use all bands.
    fwhm_reg : float, optional
        This value is converted to T and used to regularize the shapes via

            e_{1,2} = M_{1,2}/(T + T_reg)

        For Gaussians, this relationship is equivalent to smoothing by a round
        Gaussian with FWHM `fwhm_reg`.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    nband = len(mbobs)
    all_res = []
    all_psf_res = []
    all_is_shear_band = []
    all_wgts = []
    all_flags = []

    if shear_bands is None:
        shear_bands = list(range(nband))

    if fitter.kind == 'am':
        assert len(mbobs) == 1, 'Use only one band for adaptive moments'

    for band in range(nband):
        all_is_shear_band.append(True if band in shear_bands else False)
        fres = _fit_obslist(
            obslist=mbobs[band],
            fitter=fitter,
            bmask_flags=bmask_flags,
        )
        all_wgts.append(fres["wgt"])
        all_res.append(fres["obj_res"])
        all_psf_res.append(fres["psf_res"])
        all_flags.append(fres["flags"])

    return _combine_fit_results_wavg(
        all_res=all_res,
        all_psf_res=all_psf_res,
        all_is_shear_band=all_is_shear_band,
        all_wgts=all_wgts,
        model=fitter.kind,
        all_flags=all_flags,
        shear_bands=shear_bands,
        fwhm_reg=fwhm_reg,
    )


def _fit_obslist(
    *,
    obslist,
    fitter,
    bmask_flags,
):
    if len(obslist) == 0:
        # we will flag this later
        res = {}
        res["flags"] = 0
        res["flags"] |= procflags.MISSING_BAND
        res["wgt"] = 0
        res["obj_res"] = None
        res["psf_res"] = None
        return res
    else:
        return _fit_obs(
            obs=obslist[0],
            fitter=fitter,
            bmask_flags=bmask_flags,
        )


def _fit_obs(
    *,
    obs,
    fitter,
    bmask_flags,
):
    if isinstance(fitter, ngmix.prepsfmom.PrePSFMom):
        psf_go_kwargs = {"no_psf": True}
    else:
        psf_go_kwargs = {}

    res = {}
    flags = 0

    obs = symmetrize_obs_weights(obs)

    if not np.any(obs.weight > 0):
        flags |= procflags.ZERO_WEIGHTS

    if np.any((obs.bmask & bmask_flags) != 0):
        flags |= procflags.EDGE_HIT

    if flags != 0:
        # we will flag this later
        res["flags"] = flags
        res["wgt"] = 0
        res["obj_res"] = None
        res["psf_res"] = None
    else:
        res["flags"] = flags
        res["wgt"] = np.median(obs.weight[obs.weight > 0])
        res["obj_res"] = fitter.go(obs)
        res["psf_res"] = fitter.go(obs.psf, **psf_go_kwargs)

        if fitter.kind == "am" and MOMNAME == "mom":
            res["obj_res"]["mom"] = res["obj_res"]["sums"]
            res["obj_res"]["mom_cov"] = res["obj_res"]["sums_cov"]
            res["psf_res"]["mom"] = res["psf_res"]["sums"]
            res["psf_res"]["mom_cov"] = res["psf_res"]["sums_cov"]

        if res["obj_res"]["flags"] != 0:
            logger.debug("per band fitter failed: %s" % res["obj_res"]['flagstr'])

        if res["psf_res"]["flags"] != 0:
            logger.debug("per band psf fitter failed: %s" % res["psf_res"]['flagstr'])

    return res


def _sum_bands_wavg(
    *, all_res, all_is_shear_band, all_wgts, all_flags, all_wgt_res,
):
    """A function to sum all of the moments across different bands for combined moments
    estimates.

    This function sums the moments together using the entires from `all_wgts`.

        raw_mom = \\sum_i all_wgts[i] * all_res[i]["sums"]

    If `all_wgt_res` is given, then the weights are adjust to include the ratio of the
    flux moments between `all_wgt_res` and `all_res`:

        all_wgts[i] -> all_wgts[i] * all_wgt_res[i]["sums"][5] / all_res[i]["sums"][5]

    If `mom_norm` is in the results dicts in `all_res` and `all_wgt_res`, then the
    moments fields `mom` are divided by this normalization before being averaged.

    The fluxes are summed without any additional weighting beyond the entires in
    `all_wgts`

        flux = \\sum_i all_wgts[i] * all_res[i]["flux"]


    Similar sums are carried out for the the moments covariance and the flux variance
    with the approprtiate weights squared.

    Parameters
    ----------
    all_res : list of dicts
        List of the moments fit results from each band.
    all_is_shear_band : list of bool
        List of bools indicating if a given band is to be summed into the outputs and
        thus used for shear.
    all_wgts : list of floats
        List of floats giving the total weight for each band.
    all_flags : list of ints
        List of ints indicating if any of the input bands are flagged for some reason.
        These flags are ORed into the final_flags outputs.
    all_wgt_res : list of dicts, optional
        If not None, then additional weighting by the flux moment ratio of
        `all_wgt_res[i]["sums"][5] / all_res[i]["sums"][5]` is applied to the weighting
        for the moments sums. This field is used to weight the PSF moment sums by
        the fluxes of the objects so that the band-averaged sizes of stars match the
        band-averaged PSF size.

    Returns
    -------
    res : dict
        A dictionary with the following keys:

            raw_mom : np.ndarray
                The array of summed moments. You need to divide by `wgt_sum` to
                normalize the moments properly.
            raw_mom_cov : np.ndarray
                The covariance of the summed moments. You need to divide by
                `wgt_sum**2` to normalize the moments properly.
            wgt_sum : float
                The sum of the weights used in the moments sum. To get final moments,
                divide `raw_mom` by `wgt_sum` and `raw_mom_cov` by `wgt_sum**2`.
            final_flags : int
                A final set of bit flags for the summed moments. This field marks
                missing bands, missing moments, or zero weights. It is also ORed
                with all of the input `all_flags` entires.
            used_shear_bands : list of bool
                A list of which bands were used for shear. This field can be used to
                track if two successive calls to this function with slightly different
                inputs end up using the same bands.
            flux : float
                The total flux. You must divide by `wgt_sum` to get the output
                flux.
            flux_var : float
                The variance in the total flux. You must divide by `wgt_sum**2`
                to get the output flux variance.
    """
    tot_nband = len(all_res)
    raw_mom = np.zeros(6, dtype=np.float64)
    raw_mom_cov = np.zeros((6, 6), dtype=np.float64)
    wgt_sum = 0.0
    used_shear_bands = [False] * tot_nband
    final_flags = 0
    flux = 0.0
    flux_var = 0.0

    for iband, (wgt, res, issb, flags) in enumerate(zip(
        all_wgts, all_res, all_is_shear_band, all_flags
    )):
        if all_wgt_res is not None:
            wgt_res = all_wgt_res[iband]
        else:
            wgt_res = None

        if issb:
            # the input flags mark very basic failures and are ORed across all bands
            # these are things like missing and or all zero-weight data, edges, etc.
            final_flags |= flags

            # we mark missing data or moments for PSF and objects separately
            if res is None or (all_wgt_res is not None and wgt_res is None):
                final_flags |= procflags.MISSING_BAND
            elif (
                (MOMNAME not in res or MOMNAME+"_cov" not in res)
                or (
                    all_wgt_res is not None and wgt_res is not None and (
                        MOMNAME not in wgt_res or MOMNAME+"_cov" not in wgt_res
                    )
                )
            ):
                final_flags |= procflags.NOMOMENTS_FAILURE

            if (
                res is not None
                and MOMNAME+"_norm" in res
                and np.isfinite(res[MOMNAME+"_norm"])
            ):
                mom_norm = res[MOMNAME+"_norm"]
            else:
                mom_norm = 1.0

            if (
                all_wgt_res is not None
                and wgt_res is not None
                and res is not None
                and MOMNAME in res
                and MOMNAME in wgt_res
            ):
                if res[MOMNAME][5] != 0:
                    if (
                        MOMNAME+"_norm" in res
                        and np.isfinite(res[MOMNAME+"_norm"])
                        and MOMNAME+"_norm" in wgt_res
                        and np.isfinite(wgt_res[MOMNAME+"_norm"])
                    ):
                        if wgt_res[MOMNAME+"_norm"] == 0:
                            flux_mom_ratio = 1.0
                            final_flags |= procflags.ZERO_WEIGHTS
                        else:
                            flux_mom_ratio = (
                                wgt_res[MOMNAME][5]
                                / wgt_res[MOMNAME+"_norm"]
                                / res[MOMNAME][5]
                                * res[MOMNAME+"_norm"]
                            )
                    else:
                        flux_mom_ratio = wgt_res[MOMNAME][5] / res[MOMNAME][5]
                else:
                    flux_mom_ratio = 1.0
                    final_flags |= procflags.ZERO_WEIGHTS
            else:
                # we do not flag zero weight here since this is a missing band
                # or missign moments or no flux weighting
                flux_mom_ratio = 1.0

            if wgt <= 0:
                final_flags |= procflags.ZERO_WEIGHTS

            if res is not None and MOMNAME in res and MOMNAME+"_cov" in res:
                flux += (wgt * res[MOMNAME][5])
                flux_var += (wgt**2 * res[MOMNAME+"_cov"][5, 5])

                # there are a few factors here
                # wgt - the averaging weight for the moments
                # flux_mom_ratio - the ratio of the flux in wgt_res to res
                #  This factor used to properly weight PSF model averages for objects
                #  since the sums are actually things like flux * T. So for the PSF T
                #  we actually average
                #    flux_mom_ratio * sums[4] = (object flux/psf flux) * (psf_flux * T)
                #  since sums[4] = flux * T.
                # mom_norm - sum of the moments weight function
                #  This factor removes the moments dependence on area of the stamp.
                # The flux averages above do not get these factors since we want the
                # weight function to peak at 1 for flux measurements (and only care
                # about the object).

                raw_mom += (wgt * flux_mom_ratio / mom_norm * res[MOMNAME])
                raw_mom_cov += (
                    (wgt * flux_mom_ratio / mom_norm)**2
                    * res[MOMNAME+"_cov"]
                )
                wgt_sum += wgt

                used_shear_bands[iband] = True

    # make sure we flag missing data or all zero weight sums
    if sum(used_shear_bands) > 0 and wgt_sum <= 0:
        final_flags |= procflags.ZERO_WEIGHTS

    return dict(
        raw_mom=raw_mom,
        raw_mom_cov=raw_mom_cov,
        wgt_sum=wgt_sum,
        final_flags=final_flags,
        used_shear_bands=used_shear_bands,
        flux=flux,
        flux_var=flux_var,
    )


def _make_mom_res(*, raw_mom, raw_mom_cov, raw_flux, raw_flux_var, fwhm_reg):
    if fwhm_reg > 0:
        momres_t = make_mom_result(raw_mom, raw_mom_cov)

        T_reg = fwhm_to_T(fwhm_reg)

        # the moments are not normalized and are sums, so convert T_reg to a sum using
        # the flux sum first via T_reg -> T_reg * raw_mom[5]
        amat = np.eye(6)
        amat[4, 5] = T_reg

        raw_mom_orig = raw_mom.copy()
        if np.isnan(raw_mom_orig[0]):
            raw_mom[0] = 0
        if np.isnan(raw_mom_orig[1]):
            raw_mom[1] = 0
        reg_mom = np.dot(amat, raw_mom)
        if np.isnan(raw_mom_orig[0]):
            raw_mom[0] = np.nan
            reg_mom[0] = np.nan
        if np.isnan(raw_mom_orig[1]):
            raw_mom[1] = np.nan
            reg_mom[1] = np.nan

        reg_mom_cov = np.dot(amat, np.dot(raw_mom_cov, amat.T))
        momres = make_mom_result(reg_mom, reg_mom_cov)

        # use old T
        for col in ["T", "T_err", "T_flags", "T_flagstr"]:
            momres[col] = momres_t[col]
        momres["flags"] |= momres_t["flags"]
    else:
        momres = make_mom_result(raw_mom, raw_mom_cov)

    momres["flux"] = raw_flux
    if momres["flux"] <= 0:
        momres["flags"] |= ngmix.flags.NONPOS_FLUX

    if raw_flux_var > 0:
        momres["flux_err"] = np.sqrt(raw_flux_var)
        momres["s2n"] = momres["flux"] / momres["flux_err"]
    else:
        momres["flux_flags"] |= ngmix.flags.NONPOS_VAR
        momres["flux_err"] = np.nan
        momres["s2n"] = np.nan
        momres["flags"] |= ngmix.flags.NONPOS_VAR

    momres["flux_flagstr"] = procflags.get_procflags_str(momres["flux_flags"])
    momres["flagstr"] = procflags.get_procflags_str(momres["flags"])

    return momres


def _combine_fit_results_wavg(
    *, all_res, all_psf_res, all_is_shear_band, all_wgts, model, all_flags, shear_bands,
    fwhm_reg,
):
    tot_nband = len(all_res)
    nband = (
        sum(1 if issb else 0 for issb in all_is_shear_band)
        if all_is_shear_band
        else 0
    )
    nonshear_nband = (
        sum(0 if issb else 1 for issb in all_is_shear_band)
        if all_is_shear_band
        else 0
    )
    blens = [
        len(all_res), len(all_psf_res), len(all_is_shear_band),
        len(all_wgts), len(all_flags)
    ]

    n = Namer(front=model)

    data = get_wavg_output_struct(tot_nband, model, shear_bands=shear_bands)

    if (
        tot_nband == 0
        or nband == 0
        or tot_nband != nband + nonshear_nband
        or not all(b == tot_nband for b in blens)
    ):
        if nband == 0 or tot_nband == 0:
            psf_flags = procflags.MISSING_BAND
            mdet_flags = procflags.MISSING_BAND
            flux_flags = [procflags.MISSING_BAND] * tot_nband
        else:
            psf_flags = procflags.INCONSISTENT_BANDS
            mdet_flags = procflags.INCONSISTENT_BANDS
            flux_flags = [procflags.INCONSISTENT_BANDS] * tot_nband
        band_flux = [np.nan] * tot_nband
        band_flux_err = [np.nan] * tot_nband
    else:
        sum_data = _sum_bands_wavg(
            all_res=all_res,
            all_is_shear_band=all_is_shear_band,
            all_wgts=all_wgts,
            all_flags=all_flags,
            all_wgt_res=None,
        )
        mdet_flags = copy.copy(sum_data["final_flags"])

        psf_sum_data = _sum_bands_wavg(
            all_res=all_psf_res,
            all_is_shear_band=all_is_shear_band,
            all_wgts=all_wgts,
            all_flags=all_flags,
            all_wgt_res=all_res,
        )
        psf_flags = copy.copy(psf_sum_data["final_flags"])

        if (
            sum_data["final_flags"] == 0
            and psf_sum_data["final_flags"] == 0
            and sum_data["used_shear_bands"] != psf_sum_data["used_shear_bands"]
        ):
            psf_flags |= procflags.INCONSISTENT_BANDS
            mdet_flags |= procflags.INCONSISTENT_BANDS

        band_flux = []
        band_flux_err = []
        flux_flags = []
        for gres, flags in zip(all_res, all_flags):
            # the input flags mark very basic failures and are ORed across all bands
            # these are things like missing and or all zero-weight data, edges, etc.
            _flux_flags = 0
            _flux_flags |= flags

            # an object fit missing in any band is bad too
            if gres is None:
                _flux_flags |= procflags.MISSING_BAND
                band_flux.append(np.nan)
                band_flux_err.append(np.nan)
            elif (
                "flux" not in gres
                or "flux_err" not in gres
                or "flux_flags" not in gres
            ):
                _flux_flags |= procflags.NOMOMENTS_FAILURE
                band_flux.append(np.nan)
                band_flux_err.append(np.nan)
            else:
                _flux_flags |= gres["flux_flags"]
                band_flux.append(gres['flux'])
                band_flux_err.append(gres['flux_err'])
            flux_flags.append(_flux_flags)

    if psf_flags == 0:
        psf_raw_mom = psf_sum_data["raw_mom"] / psf_sum_data["wgt_sum"]
        psf_raw_mom_cov = psf_sum_data["raw_mom_cov"] / (psf_sum_data["wgt_sum"]**2)
        psf_raw_flux = psf_sum_data["flux"] / psf_sum_data["wgt_sum"]
        psf_raw_flux_var = psf_sum_data["flux_var"] / (psf_sum_data["wgt_sum"]**2)

        psf_momres = _make_mom_res(
            raw_mom=psf_raw_mom,
            raw_mom_cov=psf_raw_mom_cov,
            raw_flux=psf_raw_flux,
            raw_flux_var=psf_raw_flux_var,
            fwhm_reg=0,
        )

        psf_flags |= psf_momres["flags"]
        data[n("psf_g")] = psf_momres['e']
        data[n("psf_T")] = psf_momres['T']

    if mdet_flags == 0:
        raw_mom = sum_data["raw_mom"] / sum_data["wgt_sum"]
        raw_mom_cov = sum_data["raw_mom_cov"] / (sum_data["wgt_sum"]**2)
        raw_flux = sum_data["flux"] / sum_data["wgt_sum"]
        raw_flux_var = sum_data["flux_var"] / (sum_data["wgt_sum"]**2)

        momres = _make_mom_res(
            raw_mom=raw_mom,
            raw_mom_cov=raw_mom_cov,
            raw_flux=raw_flux,
            raw_flux_var=raw_flux_var,
            fwhm_reg=fwhm_reg,
        )

        mdet_flags |= momres["flags"]
        for col in ['s2n', 'T', 'T_err', 'T_flags']:
            data[n(col)] = momres[col]
        for col in ['e', 'e_cov']:
            data[n(col.replace('e', 'g'))] = momres[col]
        if psf_flags == 0:
            data[n('T_ratio')] = data[n('T')] / data[n('psf_T')]

    if psf_flags != 0:
        mdet_flags |= procflags.PSF_FAILURE

    if tot_nband > 1:
        data[n('band_flux')] = np.array(band_flux)
        data[n('band_flux_err')] = np.array(band_flux_err)
        data[n('band_flux_flags')] = np.array(flux_flags)
    elif tot_nband == 1:
        data[n('band_flux')] = band_flux[0]
        data[n('band_flux_err')] = band_flux_err[0]
        data[n('band_flux_flags')] = flux_flags[0]
    else:
        data[n('band_flux_flags')] = procflags.MISSING_BAND

    # now we set the flags as they would have been set in our moments code
    # any PSF failure in a shear band causes a non-zero flags value
    data[n('psf_flags')] = psf_flags
    data[n('obj_flags')] = mdet_flags
    all_flags = mdet_flags
    for f in flux_flags:
        all_flags |= f
    data[n('flags')] = all_flags

    if data[n('flags')] != 0:
        logger.debug(
            "fitter failed: flags = %s",
            procflags.get_procflags_str(data[n("flags")])
        )

    return data


def get_wavg_output_struct(nband, model, shear_bands=None):
    """
    make an output struct with default values set

    flags and psf_flags are set to NO_ATTEMPT.  The float
    fields are set to np.nan

    Parameters
    ----------
    nband: int
        Number of bands
    model: str
        The model or "kind" of fitter
    shear_bands : list of int, optional
        A list of indices into each mbobs that denotes which band is used for shear.
        If given, these are added to the output as a shear_bands field. If not given,
        this field is left empty.

    Returns
    -------
    ndarray with fields
    """
    dt = _make_combine_fit_results_wavg_dtype(
        nband=nband, model=model, shear_bands=shear_bands
    )
    data = np.zeros(1, dtype=dt)

    n = Namer(front=model)
    data[n('flags')] = procflags.NO_ATTEMPT
    data[n('psf_flags')] = procflags.NO_ATTEMPT
    data[n("obj_flags")] = procflags.NO_ATTEMPT
    data[n("T_flags")] = procflags.NO_ATTEMPT
    data[n("band_flux_flags")] = procflags.NO_ATTEMPT

    for name in data.dtype.names:
        if "flags" not in name:
            # all are float except flags
            data[name] = np.nan

    if shear_bands is not None:
        assert len(shear_bands) <= MAX_NUM_SHEAR_BANDS
        data["shear_bands"] = "".join("%s" % b for b in sorted(shear_bands))

    return data


def _make_combine_fit_results_wavg_dtype(nband, model, shear_bands):
    n = Namer(front=model)
    dt = [
        (n("flags"), 'i4'),
        (n('psf_flags'), 'i4'),
        (n('psf_g'), 'f8', 2),
        (n('psf_T'), 'f8'),
        (n("obj_flags"), 'i4'),
        (n("s2n"), "f8"),
        (n("g"), "f8", 2),
        (n("g_cov"), "f8", (2, 2)),
        (n("T"), "f8"),
        (n("T_flags"), "i4"),
        (n("T_err"), "f8"),
        (n("T_ratio"), "f8"),
    ]
    if nband > 1:
        dt += [
            (n("band_flux_flags"), 'i4', nband),
            (n("band_flux"), "f8", nband),
            (n("band_flux_err"), "f8", nband),
        ]
    else:
        dt += [
            (n("band_flux_flags"), 'i4'),
            (n("band_flux"), "f8"),
            (n("band_flux_err"), "f8"),
        ]

    dt += [("shear_bands", "U%d" % MAX_NUM_SHEAR_BANDS)]

    return dt


def symmetrize_obs_weights(obs):
    """Applies 4-fold symmetry to zero weight pixels in an observation.

    Parameters
    ----------
    obs : ngmix.Observation
        The observation to symmetrize.

    Returns
    -------
    sym_obs : ngmix.Observation
        A copy of the input observation with a symmetrized weight map.
    """
    sym_obs = obs.copy()
    if np.any(obs.weight <= 0):
        new_wgt = obs.weight.copy()
        for k in [1, 2, 3]:
            msk = np.rot90(obs.weight, k=k) <= 0
            new_wgt[msk] = 0

        with sym_obs.writeable():
            if not np.any(new_wgt > 0):
                sym_obs.ignore_zero_weight = False
            sym_obs.weight[:, :] = new_wgt

    return sym_obs


def combine_fit_res(all_res):
    """Combine fit result data structures.

    Parameters
    ----------
    all_res : list of np.ndarray
        The list of structured array results.

    Returns
    -------
    res : np.ndarray
        The combined results list.
    """

    if len(all_res) == 1:
        return all_res[0]

    dupe_cols = [
        "shear_bands",
    ]

    nobj = None
    dt = []
    for _res in all_res:
        if _res is not None:
            if nobj is None:
                nobj = _res.shape[0]
            else:
                if nobj != _res.shape[0]:
                    raise RuntimeError("All fit results must be the same length!")

            if len(dt) == 0:
                dt.extend(_res.dtype.descr)
            else:
                for descr in _res.dtype.descr:
                    if descr[0] not in dupe_cols:
                        dt.append(descr)
        else:
            if nobj is None:
                nobj = 0
            else:
                if nobj != 0:
                    raise RuntimeError(
                        "All fit results must zero length if one is None!"
                    )

    if nobj > 0:
        res = np.zeros(nobj, dtype=dt)
        for i, _res in enumerate(all_res):
            for col in _res.dtype.names:
                if col in dupe_cols:
                    if i > 0:
                        if not np.array_equal(res[col], _res[col]):
                            raise RuntimeError(
                                "Inconsistent column values "
                                "for %s when combining results!" % col
                            )
                    else:
                        res[col] = _res[col]
                else:
                    res[col] = _res[col]

        return res
    else:
        if any(_res is not None for _res in all_res):
            return np.zeros(0, dtype=dt)
        else:
            return None
