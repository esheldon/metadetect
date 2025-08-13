"""
test using lsst simple sim
"""
import sys
import numpy as np
import pytest

import logging
import ngmix
import metadetect
from metadetect import procflags
from metadetect.lsst.metadetect import run_metadetect, get_fitter
from metadetect.lsst.configs import get_config
from metadetect.lsst import util
import lsst.afw.image as afw_image
from lsst.utils import getPackageDir

try:
    getPackageDir('descwl_shear_sims')
    getPackageDir('descwl_coadd')
    skip_tests_on_simulations = False
except LookupError:
    skip_tests_on_simulations = True


ngmix_v = float(ngmix.__version__[:3])

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)


def make_lsst_sim(seed, mag=20, hlr=0.5, bands=None):
    import descwl_shear_sims

    rng = np.random.RandomState(seed=seed)
    coadd_dim = 251

    if bands is None:
        bands = ['i']

    galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=20,
        layout='grid',
        mag=mag,
        hlr=hlr,
    )

    psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = descwl_shear_sims.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        bands=bands,
    )
    return sim_data


def do_coadding(rng, sim_data, nowarp):
    from descwl_coadd.coadd import make_coadd
    from descwl_coadd.coadd_nowarp import make_coadd_nowarp

    bands = list(sim_data['band_data'].keys())

    if nowarp:
        coadd_data_list = [
            make_coadd_nowarp(
                exp=sim_data['band_data'][band][0],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                remove_poisson=False,
            )
            for band in bands
        ]
    else:
        coadd_data_list = [
            make_coadd(
                exps=sim_data['band_data'][band],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_bbox=sim_data['coadd_bbox'],
                remove_poisson=False,
            )
            for band in bands
        ]

    return util.extract_multiband_coadd_data(coadd_data_list)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
@pytest.mark.parametrize('meas_type', [None, 'wmom', 'ksigma', 'pgauss'])
@pytest.mark.parametrize('subtract_sky', [None, False, True])
@pytest.mark.parametrize("metacal_types_option", [None, "1p1m", "full"])
def test_lsst_metadetect_smoke(meas_type, subtract_sky, metacal_types_option):
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {}

    if subtract_sky is not None:
        config['subtract_sky'] = subtract_sky

    if meas_type is not None:
        config['meas_type'] = meas_type

    if metacal_types_option is not None:
        if metacal_types_option == "1p1m":
            metacal_types = ['noshear', '1p', '1m']
            config['metacal'] = {}
        elif metacal_types_option == "full":
            metacal_types = ['noshear', '1p', '1m', '2p', '2m']
            config['metacal'] = {}
        config['metacal']['types'] = metacal_types
    else:
        metacal_types = ['noshear', '1p', '1m']

    detected = afw_image.Mask.getPlaneBitMask('DETECTED')
    res = run_metadetect(rng=rng, config=config, **data)

    # we remove the DETECTED bit
    assert np.all(res['noshear']['bmask'] & detected == 0)

    for metacal_type in metacal_types:
        assert (
            metacal_type in res.keys()
        ), f"metacal_type={metacal_type} not in res.keys()"

    if meas_type is None:
        front = 'wmom'
    else:
        front = meas_type

    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in metacal_types:
        # 5x5 grid
        assert res[shear].size == 25

        assert np.any(res[shear][f"{front}_flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
@pytest.mark.skipif(ngmix_v < 2.1, reason="requires ngmix 2.1 or higher")
@pytest.mark.parametrize('meas_type', ['wmom', 'ksigma', 'pgauss'])
@pytest.mark.parametrize('fwhm_smooth', [None, 1.2])
def test_lsst_metadetect_weight(meas_type, fwhm_smooth):
    rng = np.random.RandomState(seed=882)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    fwhm = 2.0
    config = {
        'meas_type': meas_type,
        'weight': {
            'fwhm': fwhm,
        }
    }
    if meas_type != 'wmom' and fwhm_smooth is not None:
        config['weight']['fwhm_smooth'] = fwhm_smooth

    fitter = get_fitter(config=get_config(config), rng=rng)
    assert fitter.fwhm == fwhm
    if meas_type != 'wmom' and fwhm_smooth is not None:
        assert fitter.fwhm_smooth == fwhm_smooth

    res = run_metadetect(rng=rng, config=config, **data)

    gname = f'{meas_type}_g'
    flux_name = f'{meas_type}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 5x5 grid
        assert res[shear].size == 25

        assert np.any(res[shear][f"{meas_type}_flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
@pytest.mark.parametrize('fwhm_reg', [0, 0.8])
def test_lsst_metadetect_fwhm_reg(fwhm_reg):
    rng = np.random.RandomState(seed=882)

    meas_type = 'pgauss'
    bands = ['r', 'i']
    sim_data = make_lsst_sim(35, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    fwhm = 2.0
    config = {
        'meas_type': meas_type,
        'weight': {
            'fwhm': fwhm,
            'fwhm_reg': fwhm_reg,
        }
    }

    fitter = get_fitter(config=get_config(config), rng=rng)
    assert fitter.fwhm == fwhm

    res = run_metadetect(rng=rng, config=config, **data)

    gname = f'{meas_type}_g'
    flux_name = f'{meas_type}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 5x5 grid
        assert res[shear].size == 25

        assert np.any(res[shear][f"{meas_type}_flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
def test_lsst_metadetect_fwhm_reg_shapenoise():

    sdevs = {}
    for fwhm_reg in [0, 0.8]:
        meas_type = 'pgauss'
        bands = ['r', 'i']
        sim_data = make_lsst_sim(35, bands=bands)
        rng = np.random.RandomState(seed=882)
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

        fwhm = 2.0
        config = {
            'meas_type': meas_type,
            'weight': {
                'fwhm': fwhm,
                'fwhm_reg': fwhm_reg,
            }
        }

        fitter = get_fitter(config=get_config(config), rng=rng)
        assert fitter.fwhm == fwhm

        res = run_metadetect(rng=rng, config=config, **data)

        w, = np.where(res['noshear'][f'{meas_type}_flags'] == 0)

        sdevs[fwhm_reg] = res['noshear'][f'{meas_type}_g'][w, 0].std()

    assert sdevs[0.8] < 0.6 * sdevs[0]


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
def test_lsst_metadetect_am():
    rng = np.random.RandomState(seed=882)

    # only single band for am currently
    bands = ['i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    meas_type = 'am'
    config = {'meas_type': meas_type}

    res = run_metadetect(rng=rng, config=config, **data)

    gname = f'{meas_type}_g'
    flux_name = f'{meas_type}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 5x5 grid
        assert res[shear].size == 25

        assert np.any(res[shear]["am_flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        with pytest.raises(TypeError):
            len(res[shear][flux_name][0])


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
def test_lsst_metadetect_gauss():
    rng = np.random.RandomState(seed=918)

    bands = ['r', 'i', 'z']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    meas_type = 'gauss'
    config = {'meas_type': meas_type}

    res = run_metadetect(rng=rng, config=config, **data)

    gname = f'{meas_type}_g'
    flux_name = f'{meas_type}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 5x5 grid of sources
        assert res[shear].size == 25

        assert np.all(res[shear][f"{meas_type}_flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert res[shear][flux_name].shape[1] == len(bands)
        assert np.all(np.isfinite(res[shear][flux_name]))


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
def test_lsst_metadetect_fullcoadd_smoke():
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(882, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

    config = {'meas_type': 'pgauss'}
    res = run_metadetect(config=config, rng=rng, **data)

    front = 'pgauss'
    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 5x5 grid
        assert res[shear].size == 25

        assert np.any(res[shear]["pgauss_flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
def test_lsst_zero_weights(show=False):
    """
    At time of writing, DM stack will still detect in regions with inf
    variance.  Test this continues to be true.

    However, we don't have detections in BRIGHT, see test
    test_lsst_masked_as_bright
    """
    nobj = []
    seed = 55
    for do_zero in [False, True]:
        rng = np.random.RandomState(seed)
        sim_data = make_lsst_sim(seed, mag=23)
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

        if do_zero:
            data['mbexp']['i'].variance.array[50:100, 50:100] = np.inf
            data['noise_mbexp']['i'].variance.array[50:100, 50:100] = np.inf

            if show:
                import matplotlib.pyplot as mplt
                fig, axs = mplt.subplots(ncols=2)
                axs[0].imshow(data['mbexp']['i'].image.array)
                axs[1].imshow(data['mbexp']['i'].variance.array)
                mplt.show()

        resdict = run_metadetect(rng=rng, config=None, **data)

        if do_zero:
            for shear_type, tres in resdict.items():
                assert np.any(tres['wmom_flags'] & procflags.ZERO_WEIGHTS != 0)
                assert np.any(tres['wmom_psf_flags'] & procflags.NO_ATTEMPT != 0)
        else:
            for shear_type, tres in resdict.items():
                # 5x5 grid
                assert tres.size == 25

        nobj.append(resdict['noshear'].size)

    assert nobj[0] == nobj[1]


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
def test_lsst_masked_as_bright(show=False):
    """
    Make sure we don't detect in areas marked BRIGHT
    """
    seed = 55
    afw_image.Mask.addMaskPlane('BRIGHT')
    bright = afw_image.Mask.getPlaneBitMask('BRIGHT')
    for do_zero in [False, True]:
        rng = np.random.RandomState(seed)
        sim_data = make_lsst_sim(seed, mag=23)
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

        if do_zero:
            data['mbexp']['i'].variance.array[50:100, 50:100] = np.inf
            data['mbexp']['i'].mask.array[50:100, 50:100] |= bright
            data['noise_mbexp']['i'].variance.array[50:100, 50:100] = np.inf
            data['noise_mbexp']['i'].mask.array[50:100, 50:100] |= bright

        resdict = run_metadetect(rng=rng, config=None, **data)

        if show:
            import matplotlib.pyplot as mplt
            fig, axs = mplt.subplots(ncols=2)
            axs[0].imshow(data['mbexp']['i'].image.array)
            axs[1].imshow(data['mbexp']['i'].variance.array)

            axs[0].scatter(
                resdict['noshear']['col'] - resdict['noshear']['col0'],
                resdict['noshear']['row'] - resdict['noshear']['row0'],
                s=4,
                c='red',
            )
            mplt.show()

        if do_zero:
            for shear_type, tres in resdict.items():
                assert tres.size == 24
        else:
            for shear_type, tres in resdict.items():
                # 5x5 grid
                assert tres.size == 25


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
@pytest.mark.parametrize('meas_type', ['ksigma', 'pgauss'])
def test_lsst_metadetect_prepsf_stars(meas_type):
    seed = 55
    rng = np.random.RandomState(seed=seed)

    sim_data = make_lsst_sim(seed, hlr=1.0e-4, mag=23)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {'meas_type': meas_type}

    res = run_metadetect(rng=rng, config=config, **data)

    n = metadetect.util.Namer(front=meas_type)

    data = res['noshear']

    wlowT, = np.where(data[f'{meas_type}_flags'] != 0)
    wgood, = np.where(data[f'{meas_type}_flags'] == 0)

    # some will have T < 0 due to noise. Expect some with flags set
    assert wlowT.size > 0

    assert np.any((data[n('flags')][wlowT] & ngmix.flags.NONPOS_SIZE) != 0)

    assert np.any(np.isnan(data[n('g')][wlowT]))
    for field in data.dtype.names:
        if field != "shear_bands":
            assert np.all(np.isfinite(data[field][wgood])), field


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims and/or descwl_coadd not available'
)
def test_lsst_metadetect_mfrac_ormask(show=False):
    rng = np.random.RandomState(seed=116)

    ntrial = 1
    flag = 2**30

    for trial in range(ntrial):
        sim_data = make_lsst_sim(rng.randint(0, 2**30))
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

        data['mfrac_mbexp']['i'].image.array[:, :] = rng.uniform(
            size=data['mbexp']['i'].image.array.shape, low=0.2, high=0.8
        )

        for ormask in data['ormasks']:
            ormask[30:150, 30:150] = flag
            if show:
                import matplotlib.pyplot as mplt
                fig, axs = mplt.subplots(ncols=2)
                axs[0].imshow(data['mbexp']['i'].image.array)
                axs[1].imshow(ormask)
                mplt.show()

        res = run_metadetect(config=None, rng=rng, **data)

        for shear in ('noshear', '1p', '1m'):
            assert np.any(res[shear]["wmom_flags"] == 0)
            assert np.any(
                (res[shear]["mfrac"] > 0.40)
                & (res[shear]["mfrac"] < 0.60)
            )
            assert np.any(res[shear]["ormask"] & flag != 0)


if __name__ == '__main__':
    # test_lsst_metadetect_am()
    test_lsst_metadetect_gauss()
    # test_lsst_masked_as_bright(show=True)
    # test_lsst_metadetect_smoke('wmom', 'False')
    # test_lsst_metadetect_mfrac_ormask(show=True)
