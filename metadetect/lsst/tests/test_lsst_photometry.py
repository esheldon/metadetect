"""
test using lsst simple sim
"""

import sys
import numpy as np
import pytest

import logging
from metadetect.lsst import photometry as lsst_phot
from metadetect.lsst import util

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


@pytest.mark.parametrize('subtract_sky', [None, False, True])
@pytest.mark.parametrize('nowarp', [False, True])
def test_lsst_photometry_smoke(subtract_sky, nowarp):
    print('-' * 70)
    rng = np.random.RandomState(seed=116)

    bands = ['r', 'i']
    sim_data = make_lsst_sim(116, bands=bands)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=nowarp)

    config = {}

    if subtract_sky is not None:
        config['subtract_sky'] = subtract_sky

    res = lsst_phot.run_photometry(
        mbexp=data['mbexp'],
        mfrac_mbexp=data['mfrac_mbexp'],
        ormasks=data['ormasks'],
        rng=rng,
        config=config,
    )

    # 5x5 on the grid
    assert res.size == 25

    for front in ['gauss', 'pgauss']:
        if front == 'gauss':
            gname = f'{front}_g'
            assert gname in res.dtype.names

        flux_name = f'{front}_band_flux'

        assert np.any(res[f"{front}_flags"] == 0)
        assert np.all(res["mfrac"] == 0)

        assert len(res[flux_name].shape) == 2
        assert res[flux_name].shape[1] == len(bands)


if __name__ == '__main__':
    for mt in ['pgauss', 'ksigma']:
        for nowarp in [True, False]:
            test_lsst_photometry_smoke(
                meas_type=mt, subtract_sky=False, nowarp=nowarp
            )
