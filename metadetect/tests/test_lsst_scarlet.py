import time
import galsim
import pytest
import logging
import numpy as np
import ngmix

from ..lsst_measure_scarlet import (
    detect_and_deblend, measure,
)
from .. import util
from .. import vis

sim = pytest.importorskip(
    'descwl_shear_sims',
    reason='LSST codes need the descwl_shear_sims module for testing',
)

LOG = logging.getLogger('lsst_measure_scarlet')


def get_obj(rng):
    psf = galsim.Gaussian(fwhm=0.9)
    objects = []
    for i in range(3):
        obj0 = galsim.Gaussian(fwhm=1.0e-4).shift(
            dx=rng.uniform(low=-3, high=3),
            dy=rng.uniform(low=-3, high=3),
        )
        obj = galsim.Convolve(obj0, psf)
        objects.append(obj)

    return galsim.Add(objects), psf


def get_sim_data(rng, gal_type, layout):
    if gal_type == 'exp':
        coadd_dim = 151
        gal_config = {
            'mag': 23,
            'hlr': 0.5,
        }
    elif gal_type == 'wldeblend':
        # coadd_dim = 301
        coadd_dim = 151
        gal_config = None
    else:
        raise ValueError(f'bad gal type {gal_type}')

    if layout == 'pair':
        sep = 2
    else:
        sep = None

    se_dim = coadd_dim
    # bands = ['i']
    # bands = ['r', 'i', 'z']
    bands = ['g', 'r', 'i']

    galaxy_catalog = sim.galaxies.make_galaxy_catalog(
        rng=rng,
        gal_type=gal_type,
        layout=layout,
        coadd_dim=coadd_dim,
        buff=50,
        gal_config=gal_config,
        sep=sep,
    )
    psf = sim.psfs.make_fixed_psf(psf_type='gauss')

    return sim.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        bands=bands,
        g1=0.00,
        g2=0.00,
        psf=psf,
    )


def test_lsst_scarlet_smoke(
    gal_type='wldeblend',
    layout='random',
    seed=220,
    ntrial=10,
    show=False,
    loglevel='info',
):

    logging.basicConfig(level=getattr(logging, loglevel.upper()))

    print('seed:', seed)
    rng = np.random.RandomState(seed)

    tm0 = time.time()
    results = []
    for i in range(ntrial):
        LOG.info('%d/%d %g%%', i+1, ntrial, 100*(i+1)/ntrial)

        sim_data = get_sim_data(rng=rng, gal_type=gal_type, layout=layout)
        band_data = sim_data['band_data']

        exposures = [exps[0] for band, exps in band_data.items()]
        mbexp = util.get_mbexp(exposures)

        if show:
            vis.show_mbexp(mbexp, mess='original image')

        sources, detexp = detect_and_deblend(mbexp=mbexp)

        fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)
        res = measure(
            mbexp=mbexp,
            detexp=detexp,
            original_exposures=exposures,
            sources=sources,
            fitter=fitter,
            stamp_size=48,
            show=show,
        )
        results += res

    tm = time.time() - tm0

    LOG.info('time: %s', tm)
    LOG.info('time per image: %s', tm/ntrial)
    LOG.info('time per object: %s', tm/len(results))
