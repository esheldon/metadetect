import logging
import warnings
from .skysub import subtract_sky_mbobs
from . import util

from .configs import get_config
from . import measure
from . import measure_scarlet
from . import measure_shredder
from .metadetect import (
    fit_original_psfs, get_mfrac, get_fitter, get_ormask_and_bmask,
    add_original_psf, add_ormask, add_mfrac,
)

warnings.filterwarnings('ignore', category=FutureWarning)

LOG = logging.getLogger('lsst_photometry')


def run_photometry(mbobs, rng, config=None, show=False):
    """
    Run photometry on the input data

    Parameters
    ----------
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
    ndarray of results with measuremens
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
            sources, detexp = measure_scarlet.detect_and_deblend(
                mbexp=mbexp,
                thresh=config['detect']['thresh'],
                show=show,
            )
            res = measure_scarlet.measure(
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
            sources, detexp, Tvals = measure_shredder.detect_and_deblend(
                mbexp=mbexp,
                thresh=config['detect']['thresh'],
                fitter=fitter,
                stamp_size=config['stamp_size'],
                show=show,
                rng=rng,
            )
            res = measure_shredder.measure(
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

        sources, detexp = measure.detect_and_deblend(
            mbexp=mbexp,
            rng=rng,
            thresh=config['detect']['thresh'],
            show=show,
        )

        res = measure.measure(
            mbexp=mbexp,
            detexp=detexp,
            sources=sources,
            fitter=fitter,
            stamp_size=config['stamp_size'],
            find_cen=config['find_cen'],
            rng=rng,  # needed if find_cen is True
        )

    if res is not None:
        obs = mbobs[0][0]
        add_mfrac(config, mfrac, res, obs)
        add_ormask(ormask, res)
        add_original_psf(psf_stats, res)

    return res
