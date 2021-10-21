import logging
# from lsst.meas.algorithms import KernelPsf
# from lsst.afw.math import FixedKernel
# import lsst.afw.image as afw_image
from .lsst_skysub import subtract_sky_mbobs
from . import util

from .lsst_configs import get_config
from . import lsst_measure
from . import lsst_measure_scarlet
from . import lsst_measure_shredder
from .lsst_metadetect import (
    fit_original_psfs, get_mfrac, get_fitter, get_ormask_and_bmask,
    add_original_psf, add_ormask, add_mfrac,
)

LOG = logging.getLogger('lsst_photometry')


def run_photometry(mbobs, rng, config=None, show=False):
    """
    mbobs: ngmix.MultiBandObsList
        The observations to process
    rng: np.random.RandomState
        Random number generator
    config: dict, optional
        Configuration for the fitter, metacal, psf, detect, deblend, Entries
        in this dict override defaults; see lsst_configs.py
    show: bool, optional
        if set to True, images will be shown

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
    psf_stats = fit_original_psfs(
        psf_config=config['psf'], mbobs=mbobs, rng=rng,
    )

    fitter = get_fitter(config, rng=rng)

    ormask, bmask = get_ormask_and_bmask(mbobs)
    mfrac = get_mfrac(mbobs)

    # fix psf until we fix bboxes
    exposures = [obslist[0].coadd_exp for obslist in mbobs]

    mbexp = util.get_mbexp(exposures)
    if config['deblend']:
        if config['deblender'] == 'scarlet':
            sources, detexp = lsst_measure_scarlet.detect_and_deblend(
                mbexp=mbexp,
                thresh=config['detect']['thresh'],
                show=show,
            )
            res = lsst_measure_scarlet.measure(
                mbexp=mbexp,
                detexp=detexp,
                sources=sources,
                fitter=fitter,
                stamp_size=config['stamp_size'],
                rng=rng,
                show=show,
            )
        else:
            shredder_config = config['shredder_config']
            sources, detexp, Tvals = lsst_measure_shredder.detect_and_deblend(
                mbexp=mbexp,
                thresh=config['detect']['thresh'],
                fitter=fitter,
                stamp_size=config['stamp_size'],
                show=show,
                rng=rng,
            )
            res = lsst_measure_shredder.measure(
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

        for obslist in mbobs:
            assert len(obslist) == 1, 'no multiepoch'

        sources, detexp = lsst_measure.detect_and_deblend(
            mbexp=mbexp,
            rng=rng,
            thresh=config['detect']['thresh'],
            show=show,
        )

        res = lsst_measure.measure(
            mbexp=mbexp,
            detexp=detexp,
            sources=sources,
            fitter=fitter,
            stamp_size=config['stamp_size'],
        )

    if res is not None:
        obs = mbobs[0][0]
        add_mfrac(config, mfrac, res, obs)
        add_ormask(ormask, res)
        add_original_psf(psf_stats, res)

    return res
