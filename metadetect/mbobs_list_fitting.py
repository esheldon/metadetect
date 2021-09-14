"""utility functions to run fits over multiband obs lists"""
import ngmix
import esutil as eu
import numpy as np
import copy

from .util import Namer, get_ratio_var, get_ratio_error
from . import procflags


def fit_mbobs_list_nonseparable(
    *, mbobs_list, fitter, nonshear_mbobs_list=None, nonshear_fitter=None,
):
    """Fit the ojects in a list of ngmix.MultiBandObsList with a non-separable fitter.

    This function is for fitters which require a simultaneous fit across bands
    (e.g., a Gaussian fit). One fit is done for the bands for shear
    and a separate one is done for all of the bands. This second fit is
    used for fluxes only.

    Parameters
    ----------
    mbobs_list : a list of ngmix.MultiBandObsList
        The observations to use for shear measurement.
    fitter : metadetect.fitting.FitterBase or subclass
        The fitter to use for the mbobs_list
    nonshear_mbobs_list : a list of ngmix.MultiBandObsList, optional
        The list of extra observations to measure but to not combine for shear.
    fitter : metadetect.fitting.FitterBase or subclass, optional
        The fitter to use for the combined mbobs_list and nonshear_mbobs_list.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    nband = len(mbobs_list[0])

    res = fitter.go(mbobs_list)
    if nonshear_mbobs_list is not None:
        nonshear_nband = len(nonshear_mbobs_list[0])

        # build a combined mbobs list
        tot_mbobs_list = []
        for mbobs, nonshear_mbobs in zip(mbobs_list, nonshear_mbobs_list):
            _mbobs = ngmix.MultiBandObsList()
            for o in mbobs:
                _mbobs.append(o)
            for o in nonshear_mbobs:
                _mbobs.append(o)
            tot_mbobs_list.append(_mbobs)
        res_tot = nonshear_fitter.go(tot_mbobs_list)
        tot_nband = nband + nonshear_nband
    else:
        tot_nband = nband

    n = Namer(front=fitter.model)
    if tot_nband > 1:
        new_dt = [
            (n("band_flux_flags"), 'i4'),
            (n("band_flux"), "f8", tot_nband),
            (n("band_flux_err"), "f8", tot_nband),
        ]
    else:
        new_dt = [
            (n("band_flux_flags"), 'i4'),
            (n("band_flux"), "f8"),
            (n("band_flux_err"), "f8"),
        ]
    newres = eu.numpy_util.add_fields(
        res,
        new_dt,
    )
    if nonshear_mbobs_list is not None:
        newres[n("band_flux_flags")] = res_tot['flags']
        newres[n("band_flux")] = res_tot[n('flux')]
        newres[n("band_flux_err")] = res_tot[n('flux_err')]
    else:
        newres[n("band_flux_flags")] = res['flags']
        newres[n("band_flux")] = res[n('flux')]
        newres[n("band_flux_err")] = res[n('flux_err')]

    # remove the flux column
    new_dt = [
        dt
        for dt in newres.dtype.descr
        if dt[0] not in [n("flux"), n("flux_err")]
    ]
    final_res = np.zeros(newres.shape[0], dtype=new_dt)
    for c in final_res.dtype.names:
        final_res[c] = newres[c]

    return final_res


def fit_mbobs_list_separable(*, mbobs_list, fitter, nonshear_mbobs_list=None):
    """Fit the ojects in a list of ngmix.MultiBandObsList with a separable fitter.

    The fitter is run per-band and then results are combined across bands
    via a weighted average.

    Parameters
    ----------
    mbobs_list : a list of ngmix.MultiBandObsList
        The observations to use for shear measurement.
    fitter : metadetect.fitting.FitterBase or subclass
        The fitter to use per band per MultiBandObsList.
    nonshear_mbobs_list : a list of ngmix.MultiBandObsList, optional
        The list of extra observations to measure but to not combine for shear.

    Returns
    -------
    res : np.ndarray
        A structured array of the fitting results.
    """
    # run the fits
    band_res, all_is_shear_band = _run_fitter_mbobs_sep_go(
        mbobs_list=mbobs_list,
        fitter=fitter,
        nonshear_mbobs_list=nonshear_mbobs_list,
    )
    nband = len(mbobs_list[0])
    nonshear_nband = (
        len(nonshear_mbobs_list[0])
        if nonshear_mbobs_list is not None
        else 0
    )

    # now we combine via inverse variance weighting
    tot_res = np.zeros(
        len(mbobs_list),
        dtype=_make_fitter_mbobs_sep_dtype(nband, nonshear_nband, fitter.model),
    )
    for ind in range(len(mbobs_list)):
        wgts, all_bres = _extract_band_ind_res_fitter_mbobs_sep(
            ind, mbobs_list, nonshear_mbobs_list, band_res, nband, nonshear_nband
        )
        shear_mbobs = mbobs_list[ind]
        nonshear_mbobs = (
            nonshear_mbobs_list[ind]
            if nonshear_mbobs_list is not None
            else None
        )
        tot_res[ind] = _compute_wavg_fitter_mbobs_sep(
            wgts, all_bres, all_is_shear_band,
            shear_mbobs, nband, nonshear_nband,
            fitter.model,
            nonshear_mbobs=nonshear_mbobs,
        )

    if len(tot_res) == 0:
        return None
    else:
        return tot_res


def _run_fitter_mbobs_sep_go(
    *,
    mbobs_list,
    fitter,
    nonshear_mbobs_list,
):
    nband = len(mbobs_list[0])
    band_res = []
    all_is_shear_band = []

    for band in range(nband):
        all_is_shear_band.append(True)
        band_mbobs_list = []
        for mbobs in mbobs_list:
            _mbobs = ngmix.MultiBandObsList()
            _mbobs.meta = copy.deepcopy(mbobs.meta)
            _mbobs.append(mbobs[band])
            band_mbobs_list.append(_mbobs)
        band_res.append(fitter.go(band_mbobs_list))

    if nonshear_mbobs_list is not None:
        nonshear_nband = len(nonshear_mbobs_list[0])
        for band in range(nonshear_nband):
            all_is_shear_band.append(False)
            band_mbobs_list = []
            for mbobs in nonshear_mbobs_list:
                _mbobs = ngmix.MultiBandObsList()
                _mbobs.meta = copy.deepcopy(mbobs.meta)
                _mbobs.append(mbobs[band])
                band_mbobs_list.append(_mbobs)
            band_res.append(fitter.go(band_mbobs_list))

    return band_res, all_is_shear_band


def _make_fitter_mbobs_sep_dtype(nband, nonshear_nband, model):
    # combine the data by inverse variance weighted averages
    tot_nband = nband + nonshear_nband
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
        (n("T_ratio"), "f8"),
    ]
    if tot_nband > 1:
        dt += [
            (n("band_flux_flags"), 'i4'),
            (n("band_flux"), "f8", tot_nband),
            (n("band_flux_err"), "f8", tot_nband),
        ]
    else:
        dt += [
            (n("band_flux_flags"), 'i4'),
            (n("band_flux"), "f8"),
            (n("band_flux_err"), "f8"),
        ]
    return dt


def _extract_band_ind_res_fitter_mbobs_sep(
    ind, mbobs_list, nonshear_mbobs_list, band_res, nband, nonshear_nband,
):
    # extract the wgts and band results
    tot_nband = nband + nonshear_nband

    wgts = []
    all_bres = []
    for i, obslist in enumerate(mbobs_list[ind]):
        # we assume a single input image for each band
        if len(obslist) > 0:
            msk = obslist[0].weight > 0
            if not np.any(msk):
                # we will flag this later
                wgts.append(0)
            else:
                # we use the median here since that matches what was done in
                # metadetect.fitting.Moments when coadding there.
                wgts.append(np.median(obslist[0].weight[msk]))
        else:
            # we will flag this later
            wgts.append(0)
        all_bres.append(band_res[i][ind:ind+1])

    if nonshear_mbobs_list is not None:
        for i in range(nband, tot_nband):
            all_bres.append(band_res[i][ind:ind+1])
            wgts.append(1)

    wgts = np.array(wgts)
    # the weights here are for all bands for both shear and nonshear
    # measurements. we only normalize them to unity for sums over the shear
    # bands which is everything up to self.nband
    nrm = np.sum(wgts[0:nband])
    if nrm > 0:
        wgts[0:nband] = wgts[0:nband] / nrm
    else:
        wgts[0:nband] = 0
    return wgts, all_bres


def _compute_wavg_fitter_mbobs_sep(
    wgts, all_bres, all_is_shear_band, mbobs, nband, nonshear_nband, model,
    nonshear_mbobs=None,
):
    # compute the weighted averages for various columns
    tot_nband = nband + nonshear_nband
    n = Namer(front=model)
    res = np.zeros(
        1,
        dtype=_make_fitter_mbobs_sep_dtype(nband, nonshear_nband, model)
    )

    band_flux = []
    band_flux_err = []
    raw_mom = np.zeros(4, dtype=np.float64)
    raw_mom_cov = np.zeros((4, 4), dtype=np.float64)
    psf_flags = 0
    wgt_sum = 0.0
    for wgt, bres, is_shear_band in zip(wgts, all_bres, all_is_shear_band):
        res[n("band_flux_flags")] |= bres['flags'][0]

        if is_shear_band and (bres['flags'][0] & procflags.NOMOMENTS_FAILURE) == 0:
            raw_mom += (wgt * bres[n('raw_mom')][0])
            raw_mom_cov += (wgt**2 * bres[n('raw_mom_cov')][0])

            wgt_sum += wgt

            if (bres['flags'] & procflags.PSF_FAILURE) != 0:
                psf_flags |= procflags.PSF_FAILURE

            res['psf_g'] += (wgt * bres['psf_g'][0])
            res['psf_T'] += (wgt * bres['psf_T'][0])

        band_flux.append(bres[n('flux')][0])
        band_flux_err.append(bres[n('flux_err')][0])

    # now we set the flags as they would have been set in our moments code
    # any PSF failure in a shear band causes a non-zero flags value
    res['flags'] |= psf_flags
    res[n('flags')] |= psf_flags

    # we flag anything where not all of the bands have an obs or one
    # of the shear bands has zero weight
    has_all_bands = (
        all(len(obsl) > 0 for obsl in mbobs)
        and np.all(wgts[0:nband] > 0)
    )
    if nonshear_mbobs is not None:
        has_all_bands = (
            has_all_bands
            and all(len(obsl) > 0 for obsl in nonshear_mbobs)
        )

    # we need the flux > 0, flux_var > 0, T > 0 and psf measurements
    if (
        has_all_bands
        and wgt_sum > 0
        and raw_mom[0] > 0
        and raw_mom[1] > 0
        and raw_mom_cov[0, 0] > 0
        and psf_flags == 0
    ):
        raw_mom /= wgt_sum
        raw_mom_cov /= (wgt_sum**2)
        res['psf_g'] /= wgt_sum
        res['psf_T'] /= wgt_sum

        res[n('s2n')] = raw_mom[0] / np.sqrt(raw_mom_cov[0, 0])
        res[n('T')] = raw_mom[1] / raw_mom[0]
        res[n('T_err')] = get_ratio_error(
            raw_mom[1],
            raw_mom[0],
            raw_mom_cov[1, 1],
            raw_mom_cov[0, 0],
            raw_mom_cov[0, 1],
        )
        res[n('g')] = raw_mom[2:] / raw_mom[1]
        res[n('g_cov')][:, 0, 0] = get_ratio_var(
            raw_mom[2],
            raw_mom[1],
            raw_mom_cov[2, 2],
            raw_mom_cov[1, 1],
            raw_mom_cov[1, 2],
        )
        res[n('g_cov')][:, 1, 1] = get_ratio_var(
            raw_mom[3],
            raw_mom[1],
            raw_mom_cov[3, 3],
            raw_mom_cov[1, 1],
            raw_mom_cov[1, 3],
        )

        res[n('T_ratio')] = res[n('T')] / res['psf_T']
    else:
        # something above failed so mark this as a failed object
        res['flags'] |= procflags.OBJ_FAILURE
        res[n('flags')] |= procflags.OBJ_FAILURE

    if tot_nband > 1:
        res[n('band_flux')] = np.array(band_flux)
        res[n('band_flux_err')] = np.array(band_flux_err)
    else:
        res[n('band_flux')] = band_flux[0]
        res[n('band_flux_err')] = band_flux_err[0]

    return res
