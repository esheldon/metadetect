import logging
import warnings
from .skysub import subtract_sky_mbexp

from .configs import get_config
from . import measure
from .metadetect import (
    fit_original_psfs_mbexp, get_mfrac_mbexp, combine_ormasks,
    add_ormask, add_original_psf, add_mfrac,
)

warnings.filterwarnings('ignore', category=FutureWarning)

LOG = logging.getLogger('lsst_photometry')


def run_photometry(
    mbexp, rng, mfrac_mbexp=None, ormasks=None, config=None, show=False,
):
    """
    Run photometry on the input data

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    rng: np.random.RandomState
        Random number generator
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

        In the future we expect the MultibandExposure to have an ormask
        attribute
    config: dict, optional
        Configuration for the fitter, metacal, psf, detect, Entries
        in this dict override defaults; see lsst_configs.py
    show: bool, optional
        if set to True, images will be shown

    Returns
    -------
    ndarray of results with measuremens
    """

    config = get_config(config)

    ormask = combine_ormasks(mbexp, ormasks)
    mfrac, wgts = get_mfrac_mbexp(mbexp, mfrac_mbexp)

    if config['subtract_sky']:
        subtract_sky_mbexp(mbexp=mbexp, thresh=config['detect']['thresh'])

    psf_stats = fit_original_psfs_mbexp(
        mbexp=mbexp,
        wgts=wgts,
        rng=rng,
    )

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
        config=config,
        rng=rng
    )

    if res is not None:
        band = mbexp.bands[0]
        exp = mbexp[band]

        add_mfrac(config=config, mfrac=mfrac, res=res, exp=exp)
        add_ormask(ormask, res)
        add_original_psf(psf_stats, res)

    return res
