"""
test using lsst simple sim
"""
import sys
import numpy as np
import pytest

import logging
import ngmix

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)

lsst_phot = pytest.importorskip(
    'metadetect.lsst_photometry',
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
coadd_nowarp = pytest.importorskip(
    'descwl_coadd.coadd_nowarp',
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


@pytest.mark.parametrize('meas_type', [None, 'wmom', 'ksigma', 'pgauss'])
@pytest.mark.parametrize('subtract_sky', [None, False, True])
@pytest.mark.parametrize('nowarp', [False, True])
def test_lsst_photometry_smoke(meas_type, subtract_sky, nowarp):
    rng = np.random.RandomState(seed=116)

    sim_data = make_lsst_sim(116)

    if nowarp:
        coadd_obs = coadd_nowarp.make_coadd_obs_nowarp(
            exp=sim_data['band_data']['i'][0],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,
        )
    else:
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

    obslist = ngmix.ObsList()
    obslist.append(coadd_obs)
    coadd_mbobs = ngmix.MultiBandObsList()
    coadd_mbobs.append(obslist)

    config = {}

    if subtract_sky is not None:
        config['subtract_sky'] = subtract_sky

    if meas_type is not None:
        config['meas_type'] = meas_type

    res = lsst_phot.run_photometry(
        mbobs=coadd_mbobs, rng=rng,
        config=config,
    )

    # 6x6 on the grid
    assert res.size == 36

    if meas_type is None:
        front = 'wmom'
    else:
        front = meas_type

    gname = f'{front}_g'
    flux_name = f'{front}_band_flux'
    assert gname in res.dtype.names

    assert np.any(res["flags"] == 0)
    assert np.all(res["mfrac"] == 0)

    # one band
    assert len(res[flux_name].shape) == 1


@pytest.mark.parametrize('deblender', ['scarlet', 'shredder'])
def test_lsst_photometry_deblend_multiband(deblender):
    rng = np.random.RandomState(seed=99)

    bands = ('g', 'r', 'i')
    nband = len(bands)

    sim_data = make_lsst_sim(99, mag=23, bands=bands)

    coadd_mbobs = ngmix.MultiBandObsList()

    for band, exps in sim_data['band_data'].items():
        coadd_obs = coadd_nowarp.make_coadd_obs_nowarp(
            exp=exps[0],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,
        )

        # to avoid flagged edges
        coadd_obs.mfrac = np.zeros(coadd_obs.image.shape)

        obslist = ngmix.ObsList()
        obslist.append(coadd_obs)
        coadd_mbobs.append(obslist)

    config = {
        'meas_type': 'pgauss',
        'deblend': True,
        'deblender': deblender,
    }

    res = lsst_phot.run_photometry(
        mbobs=coadd_mbobs, rng=rng,
        config=config,
    )

    # 6x6 on the grid
    assert res.size == 36

    name = 'pgauss_band_flux'

    assert name in res.dtype.names

    assert len(res[name].shape) == 2
    assert len(res[name][0]) == nband


if __name__ == '__main__':
    test_lsst_photometry_deblend_multiband()
