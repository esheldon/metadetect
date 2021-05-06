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
    'descwl_shear_sims.sim',
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

    galaxy_catalog = sim.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=20,
        layout='grid',
        mag=14,
        hlr=0.5,
    )

    psf = sim.make_psf(psf_type='gauss')

    sim_data = sim.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
    )
    return sim_data


def test_lsst_metadetect_smoke():
    rng = np.random.RandomState(seed=116)

    tm0 = time.time()

    sim_data = make_lsst_sim(116)
    print("")
    mbc = coadd.MultiBandCoadds(
        rng=rng,
        interp_bright=False,
        replace_bright=False,
        data=sim_data['band_data'],
        coadd_wcs=sim_data['coadd_wcs'],
        coadd_dims=sim_data['coadd_dims'],
        psf_dims=sim_data['psf_dims'],
        byband=False,
    )

    coadd_obs = mbc.coadds['all']
    coadd_mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(coadd_obs)
    coadd_mbobs.append(obslist)

    md = lsst_metadetect.LSSTMetadetect(CONFIG, coadd_mbobs, rng)
    md.go()
    res = md.result
    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        assert np.any(res[shear]["flags"] == 0)

    total_time = time.time()-tm0
    print("time per:", total_time)
