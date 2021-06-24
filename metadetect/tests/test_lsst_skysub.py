import os
import numpy as np
import pytest

lsst_skysub_mod = pytest.importorskip(
    'metadetect.lsst_skysub',
    reason='LSST codes need the Rubin Obs. science pipelines',
)
lsst_meas_mod = pytest.importorskip(
    'metadetect.lsst_measure',
    reason='LSST codes need the Rubin Obs. science pipelines',
)
sim = pytest.importorskip(
    'descwl_shear_sims',
    reason='LSST codes need the descwl_shear_sims module for testing',
)


def make_lsst_sim(rng, gal_type, star_density=0):
    coadd_dim = 251

    # the EDGE region is 5 pixels wide but, give a bit more space because the
    # sky sub seems to fail with a lsst.pex.exceptions.wrappers.LengthError,
    # presumably due to an object near the edge

    buff = 20

    if gal_type == 'fixed':
        galaxy_catalog = sim.galaxies.FixedGalaxyCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            layout='random',
            mag=22,
            hlr=0.5,
        )
    elif gal_type == 'wldeblend':
        galaxy_catalog = sim.galaxies.WLDeblendGalaxyCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
        )
    else:
        raise ValueError('bad gal type: %s' % gal_type)

    if star_density > 0:
        stars = sim.stars.StarCatalog(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=buff,
            density=star_density,
        )
    else:
        stars = None

    psf = sim.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = sim.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        star_catalog=stars,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        se_dim=coadd_dim,
    )
    return sim_data


def make_exp(dims):
    import lsst.afw.image as afw_image
    exp = afw_image.ExposureF(width=dims[1], height=dims[0])
    exp.mask.addMaskPlane("BRIGHT")
    return exp


def check_skysub(meanvals, errvals, image_noise):

    meansky = meanvals.mean()
    stdsky = meanvals.std()
    errsky = stdsky/np.sqrt(meanvals.size)
    errmean = errvals.mean()

    # is this low enough?
    tol = image_noise / 3

    # we want the uncertainty on the mean sky to be small enough for a
    # meaningful test
    assert errsky < tol / 3

    # our sims have no sky in them
    true_skyval = 0.0

    print('true sky value is', true_skyval)
    print('mean of all trials: %g +/- %g' % (meansky, errsky))
    print('sky error from trials:', stdsky)
    print('mean predicted error:', errmean)

    assert np.abs(meansky - true_skyval) < tol


def test_skysub_smoke():
    seed = 5
    rng = np.random.RandomState(seed)

    dims = [1000, 1000]
    noise = 1.0
    skyval = 0.0

    exp = make_exp(dims)
    exp.image.array[:, :] = rng.normal(scale=noise, size=dims, loc=skyval)
    exp.variance.array[:, :] = noise**2

    lsst_skysub_mod.skysub(exp)

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

        lsst_skysub_mod.skysub(exp)

        meta = exp.getMetadata()

        meanvals[itrial] = meta['BGMEAN']
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    check_skysub(meanvals, errvals, noise)


@pytest.mark.parametrize('gal_type', ['fixed', 'wldeblend'])
def test_skysub_sim_smoke(gal_type):
    seed = 812
    rng = np.random.RandomState(seed)
    sim = make_lsst_sim(rng, gal_type=gal_type)

    exp = sim['band_data']['i'][0]
    lsst_skysub_mod.skysub(exp)

    meta = exp.getMetadata()
    assert 'BGMEAN' in meta
    assert 'BGVAR' in meta


def test_skysub_sim_fixed_gal():
    """
    check the mean sky over all trials is within 1/10 of the noise level
    """
    seed = 481
    rng = np.random.RandomState(seed)

    ntrial = 100
    meanvals = np.zeros(ntrial)
    errvals = np.zeros(ntrial)

    for itrial in range(ntrial):
        sim = make_lsst_sim(rng, gal_type='fixed')

        exp = sim['band_data']['i'][0]
        if False:
            import lsst.afw.display as afw_display
            display = afw_display.getDisplay(backend='ds9')
            display.mtv(exp)
            display.scale('log', 'minmax')

        _, _ = lsst_meas_mod.detect_and_deblend(exposure=exp, thresh=5)

        lsst_skysub_mod.skysub(exp)

        meta = exp.getMetadata()

        meanvals[itrial] = meta['BGMEAN']
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    image_noise = np.median(exp.variance.array)
    check_skysub(meanvals, errvals, image_noise)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
@pytest.mark.parametrize('star_density', [20.0])
def test_skysub_sim_wldeblend_gal(star_density):
    """
    check the mean sky over all trials is within a fraction of the noise level

    this is a slow test due to large variance in the sky determination

    putting a nominal test at stellar density of 20 but really need to do
    a full shear recover test to explore this better
    """
    seed = 213
    rng = np.random.RandomState(seed)

    ntrial = 500
    meanvals = np.zeros(ntrial)
    errvals = np.zeros(ntrial)

    for itrial in range(ntrial):
        sim = make_lsst_sim(
            rng, gal_type='wldeblend', star_density=star_density,
        )

        exp = sim['band_data']['i'][0]
        if False:
            import lsst.afw.display as afw_display
            display = afw_display.getDisplay(backend='ds9')
            display.mtv(exp)
            display.scale('log', 'minmax')

        _, _ = lsst_meas_mod.detect_and_deblend(exposure=exp, thresh=5)

        lsst_skysub_mod.skysub(exp)

        meta = exp.getMetadata()

        meanvals[itrial] = meta['BGMEAN']
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    image_noise = np.median(exp.variance.array)

    check_skysub(meanvals, errvals, image_noise)
