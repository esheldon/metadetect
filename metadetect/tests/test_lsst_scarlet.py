import galsim
import pytest
import numpy as np
import ngmix
from .. import procflags

lsst_measure_scarlet = pytest.importorskip(
    'metadetect.lsst_measure_scarlet',
    reason='LSST codes need the Rubin Obs. science pipelines',
)

sim = pytest.importorskip(
    'descwl_shear_sims',
    reason='LSST codes need the descwl_shear_sims module for testing',
)


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
    coadd_dim = 301
    if gal_type == 'exp':
        gal_config = {
            'mag': 23,
            'hlr': 0.5,
        }
    elif gal_type == 'wldeblend':
        gal_config = None
    else:
        raise ValueError(f'bad gal type {gal_type}')

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
    gal_type='exp',
    layout='grid',
    seed=220,
):

    rng = np.random.RandomState(seed)
    fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)

    sim_data = get_sim_data(rng=rng, gal_type=gal_type, layout=layout)
    band_data = sim_data['band_data']

    exposures = [exps[0] for band, exps in band_data.items()]

    lsst_measure_scarlet.detect_deblend_and_measure(
        exposures=exposures,
        fitter=fitter,
        stamp_size=48,
    )


def test_lsst_scarlet_zero_weights(
    gal_type='exp',
    layout='grid',
    seed=220,
):
    """
    Scarlet rejects sources that have zero weight, unlike SDSS deblender

    It prints linear algebra errors but continues
    """

    fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)

    nobj = []
    for do_zero in [False, True]:
        rng = np.random.RandomState(seed)

        for do_zero in [False, True]:

            sim_data = get_sim_data(rng=rng, gal_type=gal_type, layout=layout)
            band_data = sim_data['band_data']

            exposures = []
            for band, exps in band_data.items():
                exp = exps[0]
                if do_zero:
                    exp.variance.array[100:200, 100:200] = np.inf
                exposures.append(exp)

            results = lsst_measure_scarlet.detect_deblend_and_measure(
                exposures=exposures,
                fitter=fitter,
                stamp_size=48,
            )

            nobj.append(results.size)

    assert nobj[0] != nobj[1]
