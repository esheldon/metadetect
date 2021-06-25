import os
import numpy as np
import pytest
import tqdm

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


def check_skysub(meanvals, errvals, image_noise, true_sky=0):

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

    lsst_skysub_mod.determine_and_subtract_sky(exp)

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

        lsst_skysub_mod.determine_and_subtract_sky(exp)

        meta = exp.getMetadata()

        meanvals[itrial] = meta['BGMEAN']
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    check_skysub(meanvals, errvals, noise, true_sky=0)


def test_skysub_sim_smoke():
    seed = 812
    rng = np.random.RandomState(seed)
    sim = make_lsst_sim(rng, gal_type='fixed')

    exp = sim['band_data']['i'][0]
    lsst_skysub_mod.determine_and_subtract_sky(exp)

    meta = exp.getMetadata()
    assert 'BGMEAN' in meta
    assert 'BGVAR' in meta


def test_skysub_sim_fixed_gal():
    """
    check the mean sky over all trials is within 1/10 of the noise level
    """
    seed = 184
    rng = np.random.RandomState(seed)
    loglevel = 'WARN'

    ntrial = 20
    meanvals = np.zeros(ntrial)
    errvals = np.zeros(ntrial)

    for itrial in range(ntrial):
        sim = make_lsst_sim(rng, gal_type='fixed')

        exp = sim['band_data']['i'][0]

        if False:
            show_image(exp)

        # we can send subtract_sky=True but do it separately here
        # so we can see the mask before and after
        _, _ = lsst_meas_mod.detect_and_deblend(
            exposure=exp, thresh=5, loglevel=loglevel,
        )

        if False:
            show_mask(exp)

        lsst_skysub_mod.determine_and_subtract_sky(exp)

        _, _ = lsst_meas_mod.detect_and_deblend(
            exposure=exp, thresh=5, loglevel=loglevel,
        )
        if False:
            show_mask(exp)

        lsst_skysub_mod.determine_and_subtract_sky(exp)

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
    seed = 312
    rng = np.random.RandomState(seed)
    loglevel = 'WARN'

    ntrial = 20
    meanvals = np.zeros(ntrial)
    errvals = np.zeros(ntrial)

    for itrial in tqdm.trange(ntrial):
        sim = make_lsst_sim(
            rng, gal_type='wldeblend', star_density=star_density,
        )

        exp = sim['band_data']['i'][0]

        if False:
            show_image(exp)

        # we can send subtract_sky=True but do it separately here
        # so we can see the mask before and after
        _, _ = lsst_meas_mod.detect_and_deblend(
            exposure=exp, thresh=5, loglevel=loglevel,
        )
        if False:
            show_mask(exp)

        lsst_skysub_mod.determine_and_subtract_sky(exp)

        _, _ = lsst_meas_mod.detect_and_deblend(
            exposure=exp, thresh=5, loglevel=loglevel,
        )
        if False:
            show_mask(exp)

        lsst_skysub_mod.determine_and_subtract_sky(exp)

        meta = exp.getMetadata()

        meanvals[itrial] = meta['BGMEAN']
        errvals[itrial] = np.sqrt(meta['BGVAR'])

    image_noise = np.median(exp.variance.array)

    check_skysub(meanvals, errvals, image_noise)
