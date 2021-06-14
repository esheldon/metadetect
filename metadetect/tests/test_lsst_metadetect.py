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


@pytest.mark.parametrize('cls', ["LSSTMetadetect", "LSSTDeblendMetadetect"])
def test_lsst_metadetect_smoke(cls):
    rng = np.random.RandomState(seed=116)

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

    # to avoid flagged edges
    coadd_obs.mfrac = np.zeros(coadd_obs.image.shape)

    coadd_mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(coadd_obs)
    coadd_mbobs.append(obslist)

    md = getattr(lsst_metadetect, cls)(CONFIG, coadd_mbobs, rng)
    md.go()
    res = md.result
    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        assert np.any(res[shear]["flags"] == 0)
        assert np.all(res[shear]["mfrac"] == 0)


@pytest.mark.parametrize('cls', ["LSSTMetadetect", "LSSTDeblendMetadetect"])
def test_lsst_metadetect_mfrac_ormask(cls):
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
        # coadd_obs.ormask = np.ones(coadd_obs.image.shape, dtype='i4')

        coadd_mbobs = ngmix.MultiBandObsList()
        obslist = ngmix.ObsList()
        obslist.append(coadd_obs)
        coadd_mbobs.append(obslist)

        md = getattr(lsst_metadetect, cls)(CONFIG, coadd_mbobs, rng)
        md.go()
        res = md.result
        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.any(res[shear]["flags"] == 0)
            assert np.all(
                (res[shear]["mfrac"] > 0.45)
                & (res[shear]["mfrac"] < 0.55)
            )
            # assert np.all(res[shear]["ormask"] == 1)

        total_time = time.time()-tm0
        print("time per:", total_time)
