"""
test using lsst simple sim
"""
import sys
import numpy as np
import time
import pytest

import logging
import ngmix
import metadetect
from metadetect import procflags
from metadetect.lsst.metadetect import run_metadetect
from metadetect.lsst import util
import descwl_shear_sims
from descwl_coadd.coadd import make_coadd
from descwl_coadd.coadd_nowarp import make_coadd_nowarp

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)


def make_lsst_sim(seed, mag=14, hlr=0.5, bands=None):

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


@pytest.mark.parametrize('meas_type', [None, 'wmom', 'ksigma', 'pgauss'])
@pytest.mark.parametrize('subtract_sky', [None, False, True])
def test_lsst_metadetect_smoke(meas_type, subtract_sky):
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {}

    if subtract_sky is not None:
        config['subtract_sky'] = subtract_sky

    if meas_type is not None:
        config['meas_type'] = meas_type

    res = run_metadetect(rng=rng, config=config, **data)

    if meas_type is None:
        front = 'wmom'
    else:
        front = meas_type

    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 6x6 grid
        assert res[shear].size == 36

        assert np.any(res[shear]["flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


def test_lsst_metadetect_fullcoadd_smoke():
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

    config = {'meas_type': 'pgauss'}
    res = run_metadetect(config=config, rng=rng, **data)

    front = 'pgauss'
    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 6x6 grid
        assert res[shear].size == 36

        assert np.any(res[shear]["flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


def test_lsst_metadetect_find_cen():

    for itrial in (1, 2):

        if itrial == 1:
            find_cen = False
        else:
            find_cen = True

        rng = np.random.RandomState(seed=91)
        sim_data = make_lsst_sim(45, mag=23)
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

        config = {
            'meas_type': 'pgauss',
            'find_cen': find_cen,
        }

        this_res = run_metadetect(rng=rng, config=config, **data)

        if itrial == 1:
            old_res = this_res
        else:

            for shear in ('noshear', '1p', '1m'):
                assert np.any(
                    this_res[shear]['pgauss_g'] != old_res[shear]['pgauss_g']
                )


def test_lsst_metadetect_deblend_smoke():
    rng = np.random.RandomState(seed=99)

    bands = ('r', 'i')
    sim_data = make_lsst_sim(99, mag=23, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    config = {
        'meas_type': 'pgauss',
        'deblend': True,
    }

    res = run_metadetect(rng=rng, config=config, **data)

    front = 'pgauss'
    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'

    assert gname in res['noshear'].dtype.names

    for shear in ('noshear', '1p', '1m'):
        # 6x6 grid
        assert res[shear].size == 36

        assert np.any(res[shear]["flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        assert len(res[shear][flux_name].shape) == len(bands)
        assert len(res[shear][flux_name][0]) == len(bands)


def test_lsst_zero_weights(show=False):
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
                assert np.any(tres['flags'] & procflags.ZERO_WEIGHTS != 0)
                assert np.any(tres['psf_flags'] & procflags.NO_ATTEMPT != 0)
        else:
            for shear_type, tres in resdict.items():
                # 6x6 grid
                assert tres.size == 36

        nobj.append(resdict['noshear'].size)

    assert nobj[0] == nobj[1]


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

    wlowT, = np.where(data['flags'] != 0)
    wgood, = np.where(data['flags'] == 0)

    # some will have T < 0 due to noise. Expect some with flags set
    assert wlowT.size > 0

    assert np.any((data[n('flags')][wlowT] & ngmix.flags.NONPOS_SIZE) != 0)

    assert np.any(np.isnan(data[n('g')][wlowT]))
    for field in data.dtype.names:
        assert np.all(np.isfinite(data[field][wgood])), field


def test_lsst_metadetect_mfrac_ormask():
    rng = np.random.RandomState(seed=116)

    tm0 = time.time()
    ntrial = 1
    flag = 2**31

    for trial in range(ntrial):
        sim_data = make_lsst_sim(rng.randint(0, 2**30))
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=False)

        data['noise_mbexp']['i'].image.array[:, :] = rng.uniform(
            size=coadd_obs.image.shape, low=0.2, high=0.8
        )

        ormask = np.zeros(coadd_obs.image.shape, dtype='i4')
        ormask[30:150, 30:150] = flag
        coadd_obs.ormask = ormask
        # TODO make setting ormask in a CoaddObs also set exp ormask to keep
        # data consistency
        coadd_obs.coadd_exp.mask.array[:, :] = ormask

        coadd_mbobs = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(coadd_obs)
        coadd_mbobs.append(obslist)

        res = run_metadetect(mbobs=coadd_mbobs, rng=rng)

        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.any(res[shear]["flags"] == 0)
            assert np.any(
                (res[shear]["mfrac"] > 0.40)
                & (res[shear]["mfrac"] < 0.60)
            )
            assert np.any(res[shear]["ormask"] & flag != 0)

        total_time = time.time()-tm0
        print("time per:", total_time)


if __name__ == '__main__':
    test_lsst_zero_weights(show=True)
