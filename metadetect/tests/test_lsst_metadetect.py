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

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)

lsst_metadetect = pytest.importorskip(
    'metadetect.lsst_metadetect',
    reason='LSST codes need the Rubin Obs. science pipelines',
)
sim = pytest.importorskip(
    'descwl_shear_sims',
    reason='LSST codes need the descwl_shear_sims module for testing',
)
coadd = pytest.importorskip(
    'descwl_coadd.coadd',
    reason='LSST codes need the descwl_coadd module for testing',
)


def make_lsst_sim(seed, mag=14, hlr=0.5, bands=None):

    rng = np.random.RandomState(seed=seed)
    coadd_dim = 251

    if bands is None:
        bands = ['i']

    galaxy_catalog = sim.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=20,
        layout='grid',
        mag=mag,
        hlr=hlr,
    )

    psf = sim.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = sim.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        bands=bands,
    )
    return sim_data


# @pytest.mark.parametrize('meas_type', [None, 'wmom', 'ksigma', 'pgauss'])
# @pytest.mark.parametrize('subtract_sky', [None, False, True])
@pytest.mark.parametrize('meas_type', [None])
@pytest.mark.parametrize('subtract_sky', [None])
def test_lsst_metadetect_smoke(meas_type, subtract_sky):
    rng = np.random.RandomState(seed=116)

    sim_data = make_lsst_sim(116)

    print()

    coadd_obs = coadd.make_coadd_obs(
        exps=sim_data['band_data']['i'],
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_bbox=sim_data['coadd_bbox'],
        psf_dims=sim_data['psf_dims'],
        remove_poisson=False,
        rng=rng,
    )

    # to avoid flagged edges
    coadd_obs.mfrac = np.zeros(coadd_obs.image.shape)

    coadd_mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(coadd_obs)
    coadd_mbobs.append(obslist)

    config = {}

    if subtract_sky is not None:
        config['subtract_sky'] = subtract_sky

    if meas_type is not None:
        config['meas_type'] = meas_type

    res = lsst_metadetect.run_metadetect(
        mbobs=coadd_mbobs, rng=rng,
        config=config,
    )

    if meas_type is None:
        front = 'wmom'
    else:
        front = meas_type

    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'
    assert gname in res['noshear'].dtype.names

    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        assert np.any(res[shear]["flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)

        # one band
        assert len(res[shear][flux_name].shape) == 1


'''
@pytest.mark.parametrize('deblender', ['scarlet', 'shredder'])
def test_lsst_metadetect_deblend_smoke(deblender):
    rng = np.random.RandomState(seed=99)

    sim_data = make_lsst_sim(99, mag=23)

    coadd_obs = coadd.make_coadd_obs(
        exps=sim_data['band_data']['i'],
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_bbox=sim_data['coadd_bbox'],
        psf_dims=sim_data['psf_dims'],
        remove_poisson=False,
        rng=rng,
    )

    # to avoid flagged edges
    coadd_obs.mfrac = np.zeros(coadd_obs.image.shape)

    coadd_mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(coadd_obs)
    coadd_mbobs.append(obslist)

    config = {
        'meas_type': 'gap',
        'deblend': True,
        'deblender': deblender,
    }

    res = lsst_metadetect.run_metadetect(
        mbobs=coadd_mbobs, rng=rng,
        config=config,
    )

    gname = 'gap_g'

    assert gname in res['noshear'].dtype.names

    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        assert np.any(res[shear]["flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)


@pytest.mark.parametrize('deblender', ['scarlet', 'shredder'])
def test_lsst_metadetect_deblend_multiband(deblender):
    rng = np.random.RandomState(seed=99)

    bands = ('g', 'r', 'i')
    nband = len(bands)

    sim_data = make_lsst_sim(99, mag=23, bands=bands)

    coadd_mbobs = ngmix.MultiBandObsList()

    for band, exps in sim_data['band_data'].items():
        coadd_obs = coadd.make_coadd_obs(
            exps=exps,
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            remove_poisson=False,
            rng=rng,
        )
        # to avoid flagged edges
        coadd_obs.mfrac = np.zeros(coadd_obs.image.shape)

        obslist = ngmix.ObsList()
        obslist.append(coadd_obs)
        coadd_mbobs.append(obslist)

    config = {
        'meas_type': 'gap',
        'deblend': True,
        'deblender': deblender,
    }

    res = lsst_metadetect.run_metadetect(
        mbobs=coadd_mbobs, rng=rng,
        config=config,
    )

    name = 'gap_band_flux'

    assert name in res['noshear'].dtype.names

    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        assert len(res[shear][name].shape) == 2
        assert len(res[shear][name][0]) == nband


def test_lsst_zero_weights():
    nobj = []
    for do_zero in [False, True]:
        rng = np.random.RandomState(55)
        sim_data = make_lsst_sim(55, mag=23)

        coadd_obs = coadd.make_coadd_obs(
            exps=sim_data['band_data']['i'],
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            remove_poisson=False,
            rng=rng,
        )

        if do_zero:
            with coadd_obs.writeable():
                coadd_obs.weight[50:100, 50:100] = 0.0
            coadd_obs.coadd_exp.variance.array[50:100, 50:100] = np.inf

        coadd_mbobs = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(coadd_obs)
        coadd_mbobs.append(obslist)

        resdict = lsst_metadetect.run_metadetect(
            mbobs=coadd_mbobs, rng=rng, config=None,
        )

        if do_zero:

            for shear_type, tres in resdict.items():
                assert np.any(
                    tres['flags'] == (
                        procflags.OBJ_FAILURE | procflags.ZERO_WEIGHTS
                    )
                )
                assert np.any(tres['psf_flags'] == procflags.NO_ATTEMPT)

        nobj.append(resdict['noshear'].size)

    assert nobj[0] == nobj[1]


@pytest.mark.parametrize('meas_type', ['ksigma', 'pgauss'])
def test_lsst_metadetect_prepsf_stars(meas_type):
    seed = 55
    rng = np.random.RandomState(seed=seed)

    sim_data = make_lsst_sim(seed, hlr=1.0e-4, mag=23)

    coadd_obs = coadd.make_coadd_obs(
        exps=sim_data['band_data']['i'],
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_bbox=sim_data['coadd_bbox'],
        psf_dims=sim_data['psf_dims'],
        remove_poisson=False,
        rng=rng,
    )

    # to avoid flagged edges
    coadd_obs.mfrac = np.zeros(coadd_obs.image.shape)

    coadd_mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(coadd_obs)
    coadd_mbobs.append(obslist)

    config = {'meas_type': meas_type}

    res = lsst_metadetect.run_metadetect(
        mbobs=coadd_mbobs, rng=rng,
        config=config,
    )

    n = metadetect.util.Namer(front=meas_type)

    data = res['noshear']

    wlowT, = np.where(data['flags'] != 0)
    wgood, = np.where(data['flags'] == 0)

    # some will have T < 0 due to noise. Expect some with flags set
    assert wlowT.size > 0

    # TODO need to name these flags in ngmix
    assert np.all(data[n('flags')][wlowT] == 0x8)

    for field in ['s2n', 'g', 'T_ratio']:
        assert np.all(np.isnan(data[n(field)][wlowT]))

    for field in data.dtype.names:
        assert np.all(np.isfinite(data[field][wgood]))


def test_lsst_metadetect_mfrac_ormask():
    rng = np.random.RandomState(seed=116)

    tm0 = time.time()
    ntrial = 1

    for trial in range(ntrial):
        sim_data = make_lsst_sim(116)

        print("")
        coadd_obs = coadd.make_coadd_obs(
            exps=sim_data['band_data']['i'],
            coadd_wcs=sim_data['coadd_wcs'],
            coadd_bbox=sim_data['coadd_bbox'],
            psf_dims=sim_data['psf_dims'],
            remove_poisson=False,
            rng=rng,
        )
        coadd_obs.mfrac = rng.uniform(
            size=coadd_obs.image.shape, low=0.2, high=0.8
        )
        coadd_obs.ormask = np.ones(coadd_obs.image.shape, dtype='i4')

        coadd_mbobs = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(coadd_obs)
        coadd_mbobs.append(obslist)

        res = lsst_metadetect.run_metadetect(
            mbobs=coadd_mbobs, rng=rng,
        )

        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.any(res[shear]["flags"] == 0)
            print(res[shear]["mfrac"].min(), res[shear]["mfrac"].max())
            assert np.all(
                (res[shear]["mfrac"] > 0.40)
                & (res[shear]["mfrac"] < 0.60)
            )
            assert np.all(res[shear]["ormask"] == 1)

        total_time = time.time()-tm0
        print("time per:", total_time)
'''


if __name__ == '__main__':
    test_lsst_metadetect_smoke(
        # meas_type='wmom',
        # subtract_sky=False,
        meas_type=None,
        subtract_sky=None,
    )
    # test_lsst_metadetect_prepsf_stars('gap')
    # test_lsst_zero_weights()
    # test_lsst_metadetect_deblend_multiband('scarlet')
