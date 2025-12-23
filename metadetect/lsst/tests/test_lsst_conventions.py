"""
test using lsst simple sim
"""
import pytest
import sys
import numpy as np
import logging
from metadetect.lsst.configs import get_config
import metadetect.lsst.measure
from metadetect.lsst import util

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)


def make_lsst_sim(rng, g1, g2, mag=20, hlr=1.0):
    import descwl_shear_sims

    coadd_dim = 251

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
        g1=g1,
        g2=g2,
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


@pytest.mark.parametrize('ginput', [
    (0.1, 0.0),
    (0.1, 0.1),
    (0.1, -0.1),
    (0.0, 0.1),
    (0.0, -0.1),
    (-0.1, 0.0),
    (-0.1, 0.1),
    (-0.1, -0.1),
])
def test_convention(ginput):
    rng = np.random.RandomState(35)

    g1in, g2in = ginput

    sim_data = make_lsst_sim(rng, g1=g1in, g2=g2in)

    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    sources, detexp = metadetect.lsst.measure.detect_and_deblend(
        mbexp=data['mbexp'], rng=rng,
    )

    config = get_config()

    res = metadetect.lsst.measure.measure(
        mbexp=data['mbexp'],
        detexp=detexp,
        sources=sources,
        config=config,
        rng=rng,
    )

    w, = np.where(res['gauss_flags'] == 0)

    g1, g2 = res['gauss_g'][w].mean(axis=0)

    if g1in > 0:
        assert g1 > 0
    if g1in < 0:
        assert g1 < 0
    if g2in > 0:
        assert g2 > 0
    if g2in < 0:
        assert g2 < 0
