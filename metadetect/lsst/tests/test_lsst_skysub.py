import sys
import os
import numpy as np
import pytest
import tqdm
import logging
import metadetect.lsst.skysub as lsst_skysub
import metadetect.lsst.measure as lsst_measure
from lsst.utils import getPackageDir

try:
    getPackageDir('descwl_shear_sims')
    skip_tests_on_simulations = False
except LookupError:
    skip_tests_on_simulations = True

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARN,
)


def make_lsst_sim(rng, gal_type, sky_n_sigma, star_density=0):
    import descwl_shear_sims

    coadd_dim = 251

    # the EDGE region is 5 pixels wide but, give a bit more space because the
    # sky sub seems to fail with a lsst.pex.exceptions.wrappers.LengthError,
    # presumably due to an object near the edge

    buff = 20

    if gal_type == 'fixed':
        galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout='random',
            mag=22,
            hlr=0.5,
        )
    elif gal_type == 'wldeblend':
        galaxy_catalog = descwl_shear_sims.galaxies.WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
        )
    else:
        raise ValueError('bad gal type: %s' % gal_type)

    if star_density > 0:
        stars = descwl_shear_sims.stars.StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=star_density,
        )
    else:
        stars = None

    psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = descwl_shear_sims.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=stars,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        se_dim=coadd_dim,
        sky_n_sigma=sky_n_sigma,
    )
    return sim_data


def make_exp(dims):
    import lsst.afw.image as afw_image
    exp = afw_image.ExposureF(width=dims[1], height=dims[0])
    return exp


def show_mask(exp):
    import lsst.afw.display as afw_display
    display = afw_display.getDisplay(backend='ds9')
    display.mtv(exp.mask)
    input('hit a key')


def show_image(exp):
    import lsst.afw.display as afw_display
    display = afw_display.getDisplay(backend='ds9')
    display.mtv(exp)
    display.scale('log', 'minmax')
    input('hit a key')


def check_skysub(meanvals, errvals, image_noise, true_sky):

    meansky = meanvals.mean()
    stdsky = meanvals.std()
    errsky = stdsky/np.sqrt(meanvals.size)
    errmean = errvals.mean()

    tol = image_noise / 10

    # we want the uncertainty on the mean sky to be small enough for a
    # meaningful test
    assert errsky < tol / 3

    # our sims have no sky in them

    print('image_noise:', image_noise)
    print('tol:', tol)
    print('true sky value is', true_sky)
    print('mean of all trials: %g +/- %g' % (meansky, errsky))
    print('sky error from trials:', stdsky)
    print('mean predicted error:', errmean)

    assert np.abs(meansky - true_sky) < tol


def test_skysub_smoke():
    seed = 5
    rng = np.random.RandomState(seed)

    dims = [1000, 1000]
    noise = 1.0
    skyval = 0.0

    exp = make_exp(dims)
    exp.image.array[:, :] = rng.normal(scale=noise, size=dims, loc=skyval)
    exp.variance.array[:, :] = noise**2

    lsst_skysub.determine_and_subtract_sky(exp)

    meta = exp.getMetadata()
    assert 'BGMEAN' in meta
    assert 'BGVAR' in meta


def test_skysub_pure_noise():
    """
    check the mean sky over all trials is within a fraction of the noise level
    """
    seed = 5
    rng = np.random.RandomState(seed)

    dims = [1000, 1000]
    noise = 1.0
    skyval = 0.0

    ntrial = 100
    meanvals = np.zeros(ntrial)
    errvals = np.zeros(ntrial)

    for itrial in range(ntrial):
        exp = make_exp(dims)
        exp.image.array[:, :] = rng.normal(scale=noise, size=dims, loc=skyval)
        exp.variance.array[:, :] = noise**2

        lsst_skysub.determine_and_subtract_sky(exp)

        meta = exp.getMetadata()

        meanvals[itrial] = meta['BGMEAN']
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    check_skysub(meanvals, errvals, noise, true_sky=0)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
def test_skysub_sim_smoke():
    seed = 812
    rng = np.random.RandomState(seed)
    sim = make_lsst_sim(rng, gal_type='fixed', sky_n_sigma=3)

    exp = sim['band_data']['i'][0]
    lsst_skysub.determine_and_subtract_sky(exp)

    meta = exp.getMetadata()
    assert 'BGMEAN' in meta
    assert 'BGVAR' in meta


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
@pytest.mark.parametrize('sky_n_sigma', [-0.5, -2.0, -100.0])
def test_skysub_sim_fixed_gal(sky_n_sigma):
    """
    check the measured mean sky over all trials is within 1/10 of the noise
    level
    """
    seed = 184
    rng = np.random.RandomState(seed)

    ntrial = 20
    meanvals = np.zeros(ntrial)
    errvals = np.zeros(ntrial)

    for itrial in range(ntrial):
        sim = make_lsst_sim(rng, gal_type='fixed', sky_n_sigma=sky_n_sigma)

        exp = sim['band_data']['i'][0]

        noise = np.sqrt(np.median(exp.variance.array))
        true_sky = sky_n_sigma * noise

        if False:
            show_image(exp)

        lsst_skysub.iterate_detection_and_skysub(
            exposure=exp, thresh=5,
        )
        meta = exp.getMetadata()
        if 'BGMEAN' not in meta:
            raise RuntimeError('sky sub failed')

        sky_meas = meta['BGMEAN']

        meanvals[itrial] = sky_meas
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    image_noise = np.median(exp.variance.array)
    check_skysub(meanvals, errvals, image_noise, true_sky=true_sky)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize('star_density', [20.0])
@pytest.mark.parametrize('sky_n_sigma', [-0.5, -2.0, -100.0])
def test_skysub_sim_wldeblend_gal(star_density, sky_n_sigma):
    """
    check the measured mean sky over all trials is within 1/10 of the noise
    level

    putting a nominal test at stellar density of 20 but really need to do a
    full shear recover test to explore this better
    """
    seed = 312
    rng = np.random.RandomState(seed)

    ntrial = 20
    meanvals = np.zeros(ntrial)
    errvals = np.zeros(ntrial)

    for itrial in tqdm.trange(ntrial):
        sim = make_lsst_sim(
            rng, gal_type='wldeblend', star_density=star_density,
            sky_n_sigma=sky_n_sigma,
        )

        exp = sim['band_data']['i'][0]

        noise = np.sqrt(np.median(exp.variance.array))
        true_sky = sky_n_sigma * noise

        if False:
            show_image(exp)

        if True:
            lsst_skysub.iterate_detection_and_skysub(
                exposure=exp, thresh=5,
            )
            meta = exp.getMetadata()
            sky_meas = meta['BGMEAN']
        else:
            # this one is for debugging; we do the iterations ourselves so we
            # can display the result
            _, _ = lsst_measure.detect_and_deblend(
                exposure=exp, thresh=5,
            )
            if False:
                show_mask(exp)

            lsst_skysub.determine_and_subtract_sky(exp)
            sky_meas = exp.getMetadata()['BGMEAN']

            _, _ = lsst_measure.detect_and_deblend(
                exposure=exp, thresh=5,
            )
            if False:
                show_mask(exp)

            lsst_skysub.determine_and_subtract_sky(exp)

            meta = exp.getMetadata()
            sky_meas += meta['BGMEAN']

        meanvals[itrial] = sky_meas
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    image_noise = np.median(exp.variance.array)

    check_skysub(meanvals, errvals, image_noise, true_sky=true_sky)
