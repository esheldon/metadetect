import logging
import warnings
from .skysub import subtract_sky_mbexp

from .configs import get_config
from . import measure
from .metadetect import (
    fit_original_psfs_mbexp, get_mfrac_mbexp, combine_ormasks,
    add_ormask, add_original_psf, add_mfrac,
)
from lsst.meas.extensions.scarlet import ScarletDeblendTask
from .defaults import DEFAULT_THRESH
from . import util

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
        config=config,
        mbexp=mbexp,
        wgts=wgts,
        rng=rng,
    )

    dbtask = get_detect_and_deblend_task(
        thresh=config['detect']['thresh'],
        rng=rng,
    )
    sources, detexp, model_data = dbtask.run(
        mbexp=mbexp,
        show=show,
    )

    res = measure.measure(
        mbexp=mbexp,
        detexp=detexp,
        sources=sources,
        model_data=model_data,
        meas_task=dbtask.meas,
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


def get_detect_and_deblend_task(
    rng=None,
    thresh=DEFAULT_THRESH,
    deblender=None,
    config=None,
):
    """
    run detection and deblending of peaks, as well as basic measurments such as
    centroid.  The SDSS deblender is run in order to split footprints.

    We must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    rng: np.random.RandomState
        Random number generator for noise replacer
    thresh: float, optional
        The detection threshold in units of the sky noise
    config: dict, optional
        The configuration dictionary to override the defaults with.

    Returns
    -------
    sources, detexp
        The sources and the detection exposure
    """
    if deblender is not None:
        LOG.warning(
            "'deblender' kwargs is not used and will be removed soon. "
            "Specify the deblender via the config kwarg instead."
        )

    config_override = config if config is not None else {}
    if thresh:
        if 'detect' not in config_override:
            config_override['detect'] = {}
        config_override['detect']['thresholdValue'] = thresh

    config = measure.DetectAndDeblendConfig()
    config.setDefaults()

    if config_override.get('deblend', {}).pop('name', '') == "scarlet":
        config.deblend.retarget(ScarletDeblendTask)

    util.override_config(config, config_override)

    config.freeze()
    config.validate()
    task = measure.DetectAndDeblendTask(config=config)
    if rng is not None:
        task.rng = rng
    return task
