import logging
import numpy as np

import ngmix
from ngmix.gexceptions import BootPSFFailure
from ngmix.defaults import DEFAULT_LM_PARS
from ngmix.moments import make_mom_result

from .util import Namer
from . import procflags

logger = logging.getLogger(__name__)


def get_coellip_ngauss(name):
    ngauss = int(name[7:])
    return ngauss


def fit_all_psfs(mbobs, psf_conf, rng):
    """
    measure all psfs in the input observations and store the results
    in the meta dictionary, and possibly as a gmix for model fits

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to fit
    psf_conf: dict
        Config for  the measurements/fitting
    rng: np.random.RandomState
        The random number generator, used for guessers
    """
    if 'coellip' in psf_conf['model']:
        ngauss = get_coellip_ngauss(psf_conf['model'])
        fitter = ngmix.fitting.CoellipFitter(
            ngauss=ngauss,
            fit_pars=psf_conf.get('lm_pars', DEFAULT_LM_PARS),
        )
        guesser = ngmix.guessers.CoellipPSFGuesser(
            rng=rng, ngauss=ngauss,
        )
    elif psf_conf['model'] == 'wmom':
        fitter = ngmix.gaussmom.GaussMom(fwhm=psf_conf['weight_fwhm'])
        guesser = None
    elif psf_conf['model'] == 'admom':
        fitter = ngmix.admom.AdmomFitter(rng=rng)
        guesser = ngmix.guessers.GMixPSFGuesser(
            rng=rng, ngauss=1, guess_from_moms=True,
        )
    else:
        fitter = ngmix.fitting.Fitter(
            model=psf_conf['model'],
            fit_pars=psf_conf.get('lm_pars', DEFAULT_LM_PARS),
        )
        guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)

    runner = ngmix.runners.PSFRunner(
        fitter=fitter, guesser=guesser, ntry=psf_conf.get('ntry', 1),
    )

    for obslist in mbobs:
        assert len(obslist) == 1, 'metadetect is not multi-epoch'

        obs = obslist[0]
        runner.go(obs=obs)

        flags = obs.psf.meta['result']['flags']
        if flags != 0:
            raise BootPSFFailure("failed to measure psfs: %s" % flags)


def fit_mbobs_list_wavg(*, mbobs_list, fitter, bmask_flags, nonshear_mbobs_list=None):
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
    nonshear_mbobs_list : a list of ngmix.MultiBandObsList, optional
        The list of extra observations to measure but to not combine for shear.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    res = []
    for i, mbobs in enumerate(mbobs_list):
        if nonshear_mbobs_list is not None:
            nonshear_mbobs = nonshear_mbobs_list[i]
        else:
            nonshear_mbobs = None

        _res = fit_mbobs_wavg(
            mbobs=mbobs,
            fitter=fitter,
            bmask_flags=bmask_flags,
            nonshear_mbobs=nonshear_mbobs,
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
    nonshear_mbobs=None,
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
    nonshear_mbobs : ngmix.MultiBandObsList, optional
        The list of extra observations to measure but to not combine for shear.

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

    for band in range(nband):
        all_is_shear_band.append(True)
        fres = _fit_obslist(
            obslist=mbobs[band],
            fitter=fitter,
            bmask_flags=bmask_flags,
        )
        all_wgts.append(fres["wgt"])
        all_res.append(fres["obj_res"])
        all_psf_res.append(fres["psf_res"])
        all_flags.append(fres["flags"])

    if nonshear_mbobs is not None:
        nonshear_nband = len(nonshear_mbobs)
        for band in range(nonshear_nband):
            all_is_shear_band.append(False)

            fres = _fit_obslist(
                obslist=nonshear_mbobs[band],
                fitter=fitter,
                bmask_flags=bmask_flags,
            )
            # this weight is ignored anyways so set to 1
            all_wgts.append(1 if fres["wgt"] > 0 else 0)
            all_res.append(fres["obj_res"])
            all_psf_res.append(fres["psf_res"])
            all_flags.append(fres["flags"])

    # the weights here are for all bands for both shear and nonshear
    # measurements. we only normalize them to unity for sums over the shear
    # bands which is everything up to self.nband
    all_wgts = np.array(all_wgts)
    nrm = np.sum(all_wgts[0:nband])
    if nrm > 0:
        all_wgts[0:nband] = all_wgts[0:nband] / nrm
    else:
        all_wgts[0:nband] = 0

    return _combine_fit_results_wavg(
        all_res=all_res,
        all_psf_res=all_psf_res,
        all_is_shear_band=all_is_shear_band,
        all_wgts=all_wgts,
        model=fitter.kind,
        all_flags=all_flags,
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
        res["flags"] |= procflags.NO_DATA
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
        # we use the median here since that matches what was done in
        # metadetect.fitting.Moments when coadding there.
        res["flags"] = flags
        res["wgt"] = np.median(obs.weight[obs.weight > 0])
        res["obj_res"] = fitter.go(obs)
        res["psf_res"] = fitter.go(obs.psf, **psf_go_kwargs)

        if res["obj_res"]["flags"] != 0:
            logger.debug("per band fitter failed: %s" % res["obj_res"]['flagstr'])

        if res["psf_res"]["flags"] != 0:
            logger.debug("per band psf fitter failed: %s" % res["psf_res"]['flagstr'])

    return res


def _combine_fit_results_wavg(
    *, all_res, all_psf_res, all_is_shear_band, all_wgts, model, all_flags,
):
    # compute the weighted averages for various columns
    tot_nband = len(all_res)
    nband = sum(1 if issb else 0 for issb in all_is_shear_band)
    nonshear_nband = sum(0 if issb else 1 for issb in all_is_shear_band)
    assert tot_nband == nband + nonshear_nband, (
        "Inconsistent number of bands for shear vs non-shear when "
        "combining fit results!"
    )

    n = Namer(front=model)
    data = np.zeros(
        1,
        dtype=_make_combine_fit_results_wavg_dtype(tot_nband, model),
    )
    for name in data.dtype.names:
        if "flags" not in name:
            data[name] = np.nan

    band_flux = []
    band_flux_err = []
    raw_mom = np.zeros(4, dtype=np.float64)
    raw_mom_cov = np.zeros((4, 4), dtype=np.float64)
    raw_psf_g = 0
    raw_psf_T = 0
    wgt_sum = 0.0
    psf_flags = 0
    mdet_flags = 0
    flux_flags = 0
    for wgt, gres, pres, issb, flags in zip(
        all_wgts, all_res, all_psf_res, all_is_shear_band, all_flags
    ):
        ###########################
        # first handle fluxes
        # the input flags mark very basic failures and are ORed across all bands
        # these are things like missing and or all zero-weight data, edges, etc.
        flux_flags |= flags

        # an object fit missing in any band is bad too
        if gres is None:
            flux_flags |= procflags.OBJ_FAILURE
            band_flux.append(np.nan)
            band_flux_err.append(np.nan)
        else:
            flux_flags |= gres["flux_flags"]
            band_flux.append(gres['flux'])
            band_flux_err.append(gres['flux_err'])

        ###########################
        # now do shear
        if issb:
            # the input flags mark very basic failures and are ORed across all bands
            # these are things like missing and or all zero-weight data, edges, etc.
            mdet_flags |= flags

            # a PSF fit failure in any shear band ruins the shear
            if pres is None:
                mdet_flags |= procflags.PSF_FAILURE
                psf_flags |= procflags.PSF_FAILURE
            else:
                psf_flags |= pres["flags"]
                if pres["flags"] != 0:
                    mdet_flags |= procflags.PSF_FAILURE

            if gres is None:
                mdet_flags |= procflags.OBJ_FAILURE

            if (
                flags == 0
                and pres is not None
                and pres["flags"] == 0
                and gres is not None
                and ("mom" in gres and "mom_cov" in gres)
            ):
                raw_mom += (wgt * gres["mom"])
                raw_mom_cov += (wgt**2 * gres["mom_cov"])

                wgt_sum += wgt

                if "g" in pres:
                    raw_psf_g += (wgt * pres['g'])
                else:
                    raw_psf_g += (wgt * pres['e'])
                raw_psf_T += (wgt * pres['T'])
            else:
                mdet_flags |= procflags.NOMOMENTS_FAILURE

    if (
        mdet_flags == 0
        and psf_flags == 0
        and wgt_sum > 0
    ):
        raw_psf_g /= wgt_sum
        raw_psf_T /= wgt_sum
        data["psf_g"] = raw_psf_g
        data["psf_T"] = raw_psf_T

        raw_mom /= wgt_sum
        raw_mom_cov /= (wgt_sum**2)
        momres = make_mom_result(raw_mom, raw_mom_cov)
        mdet_flags |= momres["flags"]
        for col in ['s2n', 'T', 'T_err']:
            data[n(col)] = momres[col]
        for col in ['e', 'e_cov']:
            data[n(col.replace('e', 'g'))] = momres[col]
        data[n('T_ratio')] = data[n('T')] / data['psf_T']
    else:
        # something above failed so mark this as a failed object
        mdet_flags |= procflags.OBJ_FAILURE

    if tot_nband > 1:
        data[n('band_flux')] = np.array(band_flux)
        data[n('band_flux_err')] = np.array(band_flux_err)
    else:
        data[n('band_flux')] = band_flux[0]
        data[n('band_flux_err')] = band_flux_err[0]

    # now we set the flags as they would have been set in our moments code
    # any PSF failure in a shear band causes a non-zero flags value
    data['psf_flags'] = psf_flags
    data[n('flags')] = mdet_flags | psf_flags
    data[n('band_flux_flags')] = flux_flags
    data['flags'] = mdet_flags | flux_flags | psf_flags

    if data['flags'] != 0:
        logger.debug(
            "fitter failed: flags = %s",
            procflags.get_procflags_str(data["flags"])
        )

    return data


def _make_combine_fit_results_wavg_dtype(nband, model):
    n = Namer(front=model)
    dt = [
        ("flags", 'i4'),
        (n("flags"), 'i4'),
        (n("s2n"), "f8"),
        (n("T"), "f8"),
        (n("T_err"), "f8"),
        (n("g"), "f8", 2),
        (n("g_cov"), "f8", (2, 2)),
        ('psf_g', 'f8', 2),
        ('psf_T', 'f8'),
        ('psf_flags', 'i4'),
        (n("T_ratio"), "f8"),
    ]
    if nband > 1:
        dt += [
            (n("band_flux_flags"), 'i4'),
            (n("band_flux"), "f8", nband),
            (n("band_flux_err"), "f8", nband),
        ]
    else:
        dt += [
            (n("band_flux_flags"), 'i4'),
            (n("band_flux"), "f8"),
            (n("band_flux_err"), "f8"),
        ]
    return dt
