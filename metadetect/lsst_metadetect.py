"""

TODO
    - more tests
    - full shear test like in /shear_meas_test/test_shear_meas.py ?
    - understand why we are trimming the psf to even dims
    - maybe make weight_fwhm not have a default, since the default
      will most likely depend on the meas_type: wmom will have a smaller
      weight function parhaps
    - more TODO are in the code, here and in lsst_measure.py
"""
import copy
import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure
# import lsst.log
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel
import lsst.afw.image as afw_image
from .lsst_measure import detect_and_measure, subtract_sky_mbobs
from . import shearpos
from .mfrac import measure_mfrac
from . import procflags
from . import fitting
from .defaults import DEFAULT_LOGLEVEL, DEFAULT_MDET_CONFIG


def run_metadetect(
    mbobs, rng, config=None, show=False, loglevel=DEFAULT_LOGLEVEL,
):
    """
        subtract_sky, etc.
    mbobs: ngmix.MultiBandObsList
        The observations to process
    rng: np.random.RandomState
        Random number generator
    config: dict, optional
        Configuration for the fitter, metacal, psf, detect, deblend,
        Entries in this dict override defaults stored in
        defaults.DEFAULT_MDET_CONFIG
    show: bool, optional
        if True images will be shown
    loglevel: str, optional
        Default 'INFO'

    Returns
    -------
    result dict
        This is keyed by shear string 'noshear', '1p', ... or None if there was
        a problem doing the metacal steps; this only happens if the setting
        metacal_psf is set to 'fitgauss' and the fitting fails
    """

    config = get_config(config)
    metacal_config = {
        'use_noise_image': True,
        'psf': config['metacal_psf']
    }
    stamp_size = get_default_stamp_size(meas_type=config['meas_type'])

    if config['subtract_sky']:
        subtract_sky_mbobs(mbobs=mbobs, thresh=config['detect_thresh'])

    # TODO we get psf stats for the entire coadd, not location dependent
    # for each object on original image
    psf_stats = fit_original_psfs(config=config, mbobs=mbobs, rng=rng)

    fitter = get_fitter(config)

    ormask, bmask = get_ormask_and_bmask(mbobs)
    mfrac = get_mfrac(mbobs)

    odict = get_all_metacal(
        metacal_config=metacal_config, mbobs=mbobs, rng=rng, show=show,
    )

    if odict is None:
        result = None
    else:
        result = {}
        for shear_str, mbobs in odict.items():
            assert len(mbobs) == 1, 'no multiband for now'
            assert len(mbobs[0]) == 1, 'no multiepoch'
            obs = mbobs[0][0]
            exposure = obs.exposure
            res = detect_and_measure(
                exposure=exposure,
                fitter=fitter,
                stamp_size=stamp_size,
                thresh=config['detect_thresh'],
                use_deblended_stamps=config['use_deblended_stamps'],
                loglevel=loglevel,
            )

            if res is not None:
                obs = mbobs[0][0]
                add_noshear_pos(config, res, shear_str, obs)
                add_mfrac(config, mfrac, res, obs)
                add_ormask(ormask, res)
                add_original_psf(psf_stats, res)

            result[shear_str] = res

    return result


def add_noshear_pos(config, res, shear_str, obs):
    """
    add unsheared positions to the input result array
    """
    rows_noshear, cols_noshear = shearpos.unshear_positions(
        res['row'],
        res['col'],
        shear_str,
        obs,  # an example for jacobian and image shape
    )
    res['row_noshear'] = rows_noshear
    res['col_noshear'] = cols_noshear


def add_mfrac(config, mfrac, res, obs):
    """
    calculate and add mfrac to the input result array
    """
    if np.any(mfrac > 0):
        # we are using the positions with the metacal shear removed for
        # this.
        res['mfrac'] = measure_mfrac(
            mfrac=mfrac,
            x=res['col_noshear'],
            y=res['row_noshear'],
            box_sizes=res['box_size'],
            obs=obs,
            fwhm=config.get('mfrac_fwhm', None),
        )
    else:
        res['mfrac'] = 0


def add_ormask(ormask, res):
    """
    copy in ormask values using the row, col positions
    """
    for i in range(res.size):
        res['ormask'][i] = ormask[
            int(res['row'][i]), int(res['col'][i]),
        ]


def add_original_psf(psf_stats, res):
    """
    copy in psf results
    """
    res['psfrec_flags'][:] = psf_stats['flags']
    res['psfrec_g'][:, 0] = psf_stats['g1']
    res['psfrec_g'][:, 1] = psf_stats['g2']
    res['psfrec_T'][:] = psf_stats['T']


def get_fitter(config):
    """
    get the fitter based on the 'fitter' input
    """

    if config['meas_type'] == 'wmom':
        fitter = ngmix.gaussmom.GaussMom(fwhm=config['weight_fwhm'])
    elif config['meas_type'] == 'ksigma':
        fitter = ngmix.ksigmamom.KSigmaMom(fwhm=config['weight_fwhm'])
    else:
        raise ValueError("bad meas_type: '%s'" % config['meas_type'])

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

                exp = copy.deepcopy(orig_obs.coadd_exp)
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


def get_ormask_and_bmask(mbobs):
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


def get_mfrac(mbobs):
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


def fit_original_psfs(config, mbobs, rng):
    """
    fit the original psfs and get the mean g1,g2,T across
    all bands

    This can fail and flags will be set, but we proceed
    """

    psf_config = {'model': 'admom', 'ntry': 4}
    try:
        fitting.fit_all_psfs(mbobs, psf_config, rng)

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


def get_config(config=None):
    """
    extract a full config, overriding defaults with the input

    Parameters
    ----------
    config: dict, optional
        Entries in this dict override defaults stored in
        defaults.DEFAULT_MDET_CONFIG

    Returns
    -------
    Full config
    """
    if config is None:
        config = {}

    for key in config:
        if key not in DEFAULT_MDET_CONFIG:
            raise ValueError('bad key in mdet config: %s' % key)

    config_new = copy.deepcopy(DEFAULT_MDET_CONFIG)
    config_new.update(copy.deepcopy(config))

    return config_new


def get_default_stamp_size(meas_type):
    """
    get defautl stamp size for the input measurement type
    """
    if meas_type == 'wmom':
        stamp_size = 32
    elif meas_type == 'ksigma':
        # TODO figure this out
        stamp_size = 64
    else:
        raise ValueError('bad meas type: %s' % meas_type)

    return stamp_size
