"""

TODO
    - when using deblended stamps, sources are replaced by noise.
      We need to make this noise field consistent for the different
      sheared images.  We have a noise field but it is used already
      for the correlated noise corrections.  We need another that
      gets treated in parallel with the the real image: an Observation
      with noise as the image and another noise field present, and
      run through the entire metacal process as if it were real data..
      This can then be sent through as the noiseImage to the
      NoiseReplacer
    - more tests
    - full shear test like in /shear_meas_test/test_shear_meas.py ?
    - understand why we are trimming the psf to even dims
    - maybe make weight_fwhm not have a default, since the default
      will most likely depend on the meas_type: wmom will have a smaller
      weight function parhaps
    - more TODO are in the code, here and in lsst_measure.py
"""
import logging
import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure

from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel
import lsst.afw.image as afw_image

from .. import fitting
from .. import shearpos
from .. import procflags

from .skysub import subtract_sky_mbobs, subtract_sky_mbexp

from .configs import get_config
from . import measure
from . import measure_scarlet
from . import measure_shredder
from .metacal_exposures import get_metacal_mbexps_fixnoise
from .util import get_integer_center, get_jacobian

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
        Configuration for the fitter, metacal, psf, detect, deblend, Entries
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

    config = get_config(config)

    ormask = combine_ormasks(mbexp, ormasks)
    # TODO do proper mfrac
    # wgts = [1.0]*len(mbexp)
    mfrac, wgts = get_mfrac_mbexp(mfrac_mbexp)

    if config['subtract_sky']:
        subtract_sky_mbexp(mbexp=mbexp, thresh=config['detect']['thresh'])

    psf_stats = fit_original_psfs_mbexp(
        mbexp=mbexp,
        wgts=wgts,
        rng=rng,
    )

    fitter = get_fitter(config, rng=rng)

    mdict, noise_mdict = get_metacal_mbexps_fixnoise(
        mbexp=mbexp, noise_mbexp=noise_mbexp,
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
            band = mcal_mbexp.filters[0]
            exp = mcal_mbexp[band]
            add_noshear_pos_exp(res=res, shear_str=shear_str, exp=exp)

            print('TODO: ADAPT ADD_FRAC TO EXPOSURE DATA')
            add_mfrac(config=config, mfrac=mfrac, res=res, exp=exp)
            add_ormask(ormask, res)
            add_original_psf(psf_stats, res)

        result[shear_str] = res

    return result


def run_metadetect_old(
    mbobs, rng, config=None, show=False,
):
    """
    Run metadetection on the input MultiBandObsList

    Note that bright object masking must be applied outside of this function.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to process
    rng: np.random.RandomState
        Random number generator
    config: dict, optional
        Configuration for the fitter, metacal, psf, detect, deblend, Entries
        in this dict override defaults; see metadetect.lsst.configs
    show: bool, optional
        if True images will be shown

    Returns
    -------
    result dict
        This is keyed by shear string 'noshear', '1p', ... or None if there was
        a problem doing the metacal steps; this only happens if the setting
        metacal_psf is set to 'fitgauss' and the fitting fails
    """

    config = get_config(config)

    if config['subtract_sky']:
        subtract_sky_mbobs(mbobs=mbobs, thresh=config['detect']['thresh'])

    # TODO we get psf stats for the entire coadd, not location dependent
    # for each object on original image
    psf_stats = fit_original_psfs_mbobs(
        psf_config=config['psf'], mbobs=mbobs, rng=rng,
    )

    fitter = get_fitter(config, rng=rng)

    ormask, bmask = get_ormask_and_bmask_mbobs(mbobs)
    mfrac = get_mfrac_mbobs(mbobs)

    odict = get_all_metacal(
        metacal_config=config['metacal'], mbobs=mbobs, rng=rng,
    )
    if odict is None:
        result = None
    else:
        result = {}
        for shear_str, mbobs in odict.items():

            res = detect_deblend_and_measure(
                mbobs=mbobs,
                fitter=fitter,
                config=config,
                rng=rng,
                show=show,
            )

            if res is not None:
                obs = mbobs[0][0]
                add_noshear_pos_obs(config, res, shear_str, obs)
                add_mfrac(config, mfrac, res, obs)
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
    user has a choice whether to use deblended postage stamps for the
    measurement.

    Parameters
    ----------
    exposure: Exposure
        Exposure on which to detect and measure
    fitter: e.g. ngmix.gaussmom.GaussMom or ngmix.ksigmamom.KSigmaMom
        For calculating moments
    thresh: float
        The detection threshold in units of the sky noise
    stamp_size: int
        Size for postage stamps.
    deblend: bool
        If True, use deblended the postage stamps for each measurement using
        the scarlet deblender.  If not True, the SDSS deblender code is used
        but only to find the sub-peaks in the footprint, and bands are coadded
    deblender: str
        Deblender to use, scarlet or shredder
    show: bool, optional
        If set to True, show images during processing
    """

    if config['deblend']:

        if config['deblender'] == 'scarlet':
            LOG.info('measuring with deblended stamps')
            sources, detexp = measure_scarlet.detect_and_deblend(
                mbexp=mbexp,
                thresh=config['detect']['thresh'],
                show=show,
            )
            results = measure_scarlet.measure(
                mbexp=mbexp,
                detexp=detexp,
                sources=sources,
                fitter=fitter,
                stamp_size=config['stamp_size'],
                rng=rng,
                show=show,
            )
        else:
            LOG.info('measuring with the Shredder')

            shredder_config = config['shredder_config']

            sources, detexp, Tvals = measure_shredder.detect_and_deblend(
                mbexp=mbexp,
                thresh=config['detect']['thresh'],
                fitter=fitter,
                stamp_size=config['stamp_size'],
                rng=rng,
                show=show,
            )
            results = measure_shredder.measure(
                mbexp=mbexp,
                detexp=detexp,
                sources=sources,
                fitter=fitter,
                stamp_size=config['stamp_size'],
                Tvals=Tvals,
                shredder_config=shredder_config,
                rng=rng,
                show=show,
            )

    else:

        LOG.info('measuring with blended stamps')

        sources, detexp = measure.detect_and_deblend(
            mbexp=mbexp,
            rng=rng,
            thresh=config['detect']['thresh'],
            show=show,
        )
        results = measure.measure(
            mbexp=mbexp,
            detexp=detexp,
            sources=sources,
            fitter=fitter,
            stamp_size=config['stamp_size'],
            find_cen=config['find_cen'],
            rng=rng,  # needed if find_cen is True
        )

    return results


def add_noshear_pos_exp(res, shear_str, exp):
    """
    add unsheared positions to the input result array
    """

    dims = exp.image.array.shape

    cen, _ = get_integer_center(
        wcs=exp.getWcs(),
        bbox=exp.getBBox(),
        as_double=True,
    )
    jac = get_jacobian(exp=exp, cen=cen)

    rows_noshear, cols_noshear = shearpos.unshear_positions(
        res['row'] - res['row0'],
        res['col'] - res['col0'],
        shear_str,
        jac=jac,
        dims=dims,
    )
    res['row_noshear'] = rows_noshear
    res['col_noshear'] = cols_noshear


def add_noshear_pos_obs(config, res, shear_str, obs):
    """
    add unsheared positions to the input result array
    """
    rows_noshear, cols_noshear = shearpos.unshear_positions_obs(
        res['row'] - res['row0'],
        res['col'] - res['col0'],
        shear_str,
        obs,  # an example for jacobian and image shape
    )
    res['row_noshear'] = rows_noshear
    res['col_noshear'] = cols_noshear


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
            x=res['col_noshear'],
            y=res['row_noshear'],
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
    obs : ngmix.Observation
        An observation that holds the weight maps, WCS Jacobian, etc
        corresponding to `mfrac`.
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
        x = int(np.floor(x[i] + 0.5))
        y = int(np.floor(y[i] + 0.5))

        xstart = x - box_rad
        xend = x + box_rad + 1
        ystart = y - box_rad
        yend = y + box_rad + 1

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
            cy, cx = (x[i] - xstart, y[i] - ystart)
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

    elif meas_type == 'em':
        fitter = None
    else:
        fwhm = config['weight']['fwhm']
        if meas_type == 'wmom':
            fitter = ngmix.gaussmom.GaussMom(fwhm=fwhm)
        elif meas_type == 'ksigma':
            fitter = ngmix.ksigmamom.KSigmaMom(fwhm=fwhm)
        elif meas_type == 'pgauss':
            fitter = ngmix.prepsfmom.PGaussMom(fwhm=fwhm)
        else:
            raise ValueError("bad meas_type: '%s'" % meas_type)

    return fitter


def get_all_metacal(metacal_config, mbobs, rng, show=False):
    """
    get the sheared versions of the observations

    call the parent and then add in the stack exposure with image copied
    in, modify the variance and set the new psf
    """

    orig_mbobs = mbobs

    try:
        odict = ngmix.metacal.get_all_metacal(
            orig_mbobs,
            rng=rng,
            **metacal_config,
        )
    except BootPSFFailure:
        # this can happen if we were using psf='fitgauss'
        return None

    # make sure .exposure is set for each obs
    for mtype, mbobs in odict.items():
        for band in range(len(mbobs)):

            obslist = mbobs[band]
            orig_obslist = orig_mbobs[band]

            for iobs in range(len(obslist)):
                obs = obslist[iobs]
                orig_obs = orig_obslist[iobs]

                # exp = copy.deepcopy(orig_obs.coadd_exp)
                exp = afw_image.ExposureF(orig_obs.coadd_exp, deep=True)
                exp.image.array[:, :] = obs.image

                # we ran fixnoise, need to update variance plane
                exp.variance.array[:, :] = exp.variance.array[:, :]*2

                psf_image = obs.psf.image
                stack_psf = KernelPsf(
                    FixedKernel(
                        afw_image.ImageD(psf_image.astype(float))
                    )
                )
                exp.setPsf(stack_psf)
                obs.exposure = exp

                if show:
                    import descwl_coadd.vis
                    descwl_coadd.vis.show_image_and_mask(exp)
                    input('hit a key')

    return odict


def combine_ormasks(mbexp, ormasks):
    if ormasks is None:
        bands = mbexp.filters
        dims = mbexp[bands[0]].image.array.shape
        ormask = np.zeros(dims, dtype='i4')
    else:
        for imask, tormask in enumerate(ormasks):
            if imask == 0:
                ormask = tormask.copy()
            else:
                ormask |= tormask

    return ormask


def get_ormask_and_bmask_mbobs(mbobs):
    """
    get the ormask and bmask, ored from all epochs
    """

    for band, obslist in enumerate(mbobs):
        nepoch = len(obslist)
        assert nepoch == 1, 'expected 1 epoch, got %d' % nepoch

        obs = obslist[0]

        if band == 0:
            ormask = obs.ormask.copy()
            bmask = obs.bmask.copy()
        else:
            ormask |= obs.ormask
            bmask |= obs.bmask

    return ormask, bmask


def get_mfrac_mbobs(mbobs):
    """
    set the masked fraction image, averaged over all bands
    """
    wgts = []
    mfrac = np.zeros_like(mbobs[0][0].image)
    for band, obslist in enumerate(mbobs):
        nepoch = len(obslist)
        assert nepoch == 1, 'expected 1 epoch, got %d' % nepoch

        obs = obslist[0]
        wgt = np.median(obs.weight)
        if hasattr(obs, "mfrac"):
            mfrac += (obs.mfrac * wgt)
        wgts.append(wgt)

    mfrac = mfrac / np.sum(wgts)
    return mfrac


def get_mfrac_mbexp(mfrac_mbexp):
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
    for iband, exp in enumerate(mfrac_mbexp):
        w = np.where(np.isfinite(exp.variance.array))
        if w[0].size == 0:
            raise ValueError('no variance are finite')

        var = np.median(exp.variance.array[w])
        wgt = 1/var
        wgts.append(wgt)
        wsum += wgt
        if iband == 0:
            mfrac = wgt * np.zeros_like(exp.image.array)
        else:
            mfrac += wgt * exp.image.array

    mfrac *= 1.0/wsum

    return mfrac, wgts


def fit_original_psfs_mbobs(psf_config, mbobs, rng):
    """
    fit the original psfs and get the mean g1,g2,T across
    all bands

    This can fail and flags will be set, but we proceed
    """

    try:
        fitting.fit_all_psfs(mbobs=mbobs, psf_conf=psf_config, rng=rng)

        g1sum = 0.0
        g2sum = 0.0
        Tsum = 0.0
        wsum = 0.0

        for obslist in mbobs:
            for obs in obslist:
                wt = obs.weight.max()
                res = obs.psf.meta['result']
                T = res['T']
                if 'e' in res:
                    g1, g2 = res['e']
                else:
                    g1, g2 = res['g']

                g1sum += g1*wt
                g2sum += g2*wt
                Tsum += T*wt
                wsum += wt

        if wsum <= 0.0:
            raise BootPSFFailure('zero weights, could not '
                                 'get mean psf properties')
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


def fit_original_psfs_mbexp(mbexp, rng, wgts):
    """
    fit the original psfs at the center of the image and get the mean g1,g2,T
    across all bands

    This can fail and flags will be set, but we proceed
    """
    from .measure import extract_psf_image

    assert len(wgts) == len(mbexp)
    wsum = sum(wgts)
    if wsum <= 0:
        raise ValueError(f'got sum(wgts) = {wsum}')

    fitter = ngmix.admom.AdmomFitter(rng=rng)
    guesser = ngmix.guessers.GMixPSFGuesser(
        rng=rng, ngauss=1, guess_from_moms=True,
    )
    runner = ngmix.runners.PSFRunner(fitter=fitter, guesser=guesser, ntry=4,)

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

            psf_im = extract_psf_image(exp, cen)

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
