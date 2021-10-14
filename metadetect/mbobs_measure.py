"""
utility functions to run measurements over a MultiBandObsList
"""
import ngmix
import numpy as np

from ngmix.ksigmamom import KSigmaMom
from ngmix.prepsfmom import PrePSFGaussMom

from . import procflags


def measure_mbobs(mbobs, fitter, nonshear_mbobs=None):
    """
    Fit the object in a ngmix.MultiBandObsList with a separable fitter.

    The fitter is run per-band and then results are combined across bands
    via a weighted average.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to use for shear measurement.
    fitter: e.g. ngmix.prepsfmom.PrePsfGaussMom
        The fitter to use per band per MultiBandObsList.
    nonshear_mbobs: ngmix.MultiBandObsList, optional
        The extra observations to measure but to not combine for shear.

    Returns
    -------
    res, psf_res: dicts
        A result dict for the object and psf.  These contain weighted averages
        over the shear bands.  The object res also has band_flux/band_flux_err
        for each band
    """

    reslist, psf_reslist = _run_band_measurements(mbobs=mbobs, fitter=fitter)
    if nonshear_mbobs is not None:
        tres, tpsf = _run_band_measurements(mbobs=mbobs, fitter=fitter)
        reslist += tres
        psf_reslist += tpsf

    wgts = _get_weights_sep(mbobs, nonshear_mbobs)

    res, psf_res = _compute_wavg_sep(
        wgts=wgts,
        reslist=reslist,
        psf_reslist=psf_reslist,
        mbobs=mbobs,
        nonshear_mbobs=nonshear_mbobs,
    )

    res['numiter'] = 1
    if 'e' in res:
        res['g'] = res['e']
        res['g_cov'] = res['e_cov']

    psf_res['numiter'] = 1
    if 'e' in psf_res:
        psf_res['g'] = res['e']
        psf_res['g_cov'] = res['e_cov']

    return res, psf_res


def _get_nband(mbobs):
    return (
        len(mbobs)
        if mbobs is not None
        else 0
    )


def _run_band_measurements(mbobs, fitter):
    is_prepsf = (
        isinstance(fitter, KSigmaMom) or isinstance(fitter, PrePSFGaussMom)
    )

    reslist = [
        fitter.go(obslist[0]) for obslist in mbobs
    ]
    if is_prepsf:
        psf_reslist = [
            fitter.go(obslist[0].psf, no_psf=True) for obslist in mbobs
        ]
    else:
        psf_reslist = [
            fitter.go(obslist[0].psf) for obslist in mbobs
        ]

    return reslist, psf_reslist


def _get_weights_sep(
    mbobs, nonshear_mbobs,
):
    """
    get the weights for each band.  Non shear bands are given a weight 1 here,
    but note those are just placeholders, they are not used for any averaging
    """
    nband = _get_nband(mbobs)
    nonshear_nband = _get_nband(nonshear_mbobs)

    wgts = []

    for obslist in mbobs:
        # we assume a single input image for each band
        if len(obslist) > 0:
            obs = obslist[0]
            w = np.where(obs.weight > 0)
            if w[0].size == 0:
                # we will flag this later
                wgts.append(0)
            else:
                # we use the median here since that matches what was done in
                # metadetect.fitting.Moments when coadding there.
                wgts.append(np.median(obs.weight[w]))
        else:
            # we will flag this later
            wgts.append(0)

    if nonshear_mbobs is not None:
        wgts += [1.0]*nonshear_nband

    wgts = np.array(wgts)

    # the weights here are for all bands for both shear and nonshear
    # measurements. we only normalize them to unity for sums over the shear
    # bands which is everything up to self.nband

    nrm = np.sum(wgts[0:nband])
    if nrm > 0:
        wgts[0:nband] = wgts[0:nband] / nrm
    else:
        wgts[0:nband] = 0

    return wgts


def _compute_wavg_sep(
    wgts, reslist, psf_reslist, mbobs, nonshear_mbobs=None,
):
    """
    compute weighted averages for shear quantities, and tabulate
    the flux/flux_err in each band
    """

    nband = _get_nband(mbobs)
    nonshear_nband = _get_nband(nonshear_mbobs)

    tot_nband = nband + nonshear_nband

    band_flux = []
    band_flux_err = []

    raw_mom = reslist[0]['mom'] * 0
    raw_cov = reslist[0]['mom_cov'] * 0

    raw_psf_mom = psf_reslist[0]['mom'] * 0
    raw_psf_cov = psf_reslist[0]['mom_cov'] * 0

    psf_flags = 0
    band_flux_flags = 0
    wgt_sum = 0.0

    for iband in range(tot_nband):
        if iband < nband:
            is_shear_band = True
        else:
            is_shear_band = False

        wgt = wgts[iband]
        bres = reslist[iband]
        psf_bres = psf_reslist[iband]

        band_flux_flags |= bres['flux_flags']

        # checking only for bad covariance. TODO need names for flags in ngmix
        if is_shear_band and (bres['flags'] & 0x40) == 0:
            raw_mom += (wgt * bres['mom'])
            raw_cov += (wgt**2 * bres['mom_cov'])

            raw_psf_mom += (wgt * psf_bres['mom'])
            raw_psf_cov += (wgt**2 * psf_bres['mom_cov'])

            wgt_sum += wgt

        psf_flags |= psf_bres['flags']

        band_flux.append(bres['flux'])
        band_flux_err.append(bres['flux_err'])

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
    # for the shear measurements
    if (
        has_all_bands
        and wgt_sum > 0
        and raw_mom[0] > 0
        and raw_mom[1] > 0
        and raw_cov[0, 0] > 0
        and psf_flags == 0
    ):
        raw_mom /= wgt_sum
        raw_cov /= (wgt_sum**2)

        raw_psf_mom /= wgt_sum
        raw_psf_cov /= (wgt_sum**2)

        # TODO make _make_mom_res public in ngmix
        res = ngmix.prepsfmom._make_mom_res(raw_mom, raw_cov)
        psf_res = ngmix.prepsfmom._make_mom_res(raw_psf_mom, raw_psf_cov)

    else:
        # something above failed so mark this as a failed object
        res = {'flags': procflags.OBJ_FAILURE}
        psf_res = {'flags': procflags.OBJ_FAILURE}

    res['band_flux_flags'] = band_flux_flags
    res['band_flux'] = band_flux
    res['band_flux_err'] = band_flux_err

    return res, psf_res
