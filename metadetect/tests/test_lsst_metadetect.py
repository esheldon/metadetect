"""
test using lsst simple sim
"""
import numpy as np
import time
import pytest

import ngmix

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

CONFIG = {
    "model": "wmom",
    "bmask_flags": 0,
    "metacal": {
        "use_noise_image": True,
        "psf": "fitgauss",
    },
    "psf": {
        "model": "gauss",
        "lm_pars": {},
        "ntry": 2,
    },
    "weight": {
        "fwhm": 1.2,
    },
    "detect": {
        "thresh": 10.0,
    },
    "subtract_sky": False,
    'meds': {},
}


def make_lsst_sim(seed):
    rng = np.random.RandomState(seed=seed)
    coadd_dim = 251

    galaxy_catalog = sim.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=20,
        layout='grid',
        mag=14,
        hlr=0.5,
    )

    psf = sim.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = sim.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )
    return sim_data


@pytest.mark.parametrize('meas_type', [None, 'wmom', 'ksigma'])
@pytest.mark.parametrize('subtract_sky', [None, False, True])
@pytest.mark.parametrize('use_deblended_stamps', [None, False, True])
def test_lsst_metadetect_smoke(meas_type, subtract_sky, use_deblended_stamps):
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

    if use_deblended_stamps is not None:
        config['use_deblended_stamps'] = use_deblended_stamps

    res = lsst_metadetect.run_metadetect(
        mbobs=coadd_mbobs, rng=rng,
        config=config,
    )

    if meas_type is None:
        gname = 'wmom_g'
    else:
        gname = '%s_g' % meas_type

    assert gname  in res['noshear'].dtype.names

    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        assert np.any(res[shear]["flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)


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
