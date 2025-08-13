import numpy as np
from metadetect.lsst import util
import lsst.afw.image as afw_image
import lsst.geom as geom
import pytest
from lsst.utils import getPackageDir

try:
    getPackageDir('descwl_shear_sims')
    skip_tests_on_simulations = False
except LookupError:
    skip_tests_on_simulations = True


def make_lsst_sim(rng, dim, mag=22, hlr=0.5, bands=['i']):
    import descwl_shear_sims

    coadd_dim = dim
    se_dim = dim

    galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=0,
        layout='grid',
        mag=mag,
        hlr=hlr,
    )

    psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = descwl_shear_sims.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        g1=0.00,
        g2=0.00,
        psf=psf,
        bands=bands,
    )
    return sim_data


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
@pytest.mark.parametrize('copy_mask_to', ['ormask', 'bmask'])
def test_conversion(copy_mask_to):
    seed = 9911
    dim = 200
    rng = np.random.RandomState(seed=seed)

    sim_data = make_lsst_sim(rng, dim)
    exp = sim_data['band_data']['i'][0]

    cen_integer, _ = util.get_integer_center(
        wcs=exp.getWcs(),
        bbox=exp.getBBox(),
    )
    cen = geom.Point2D(cen_integer)
    psf_obj = exp.getPsf()
    psf_image = psf_obj.computeKernelImage(cen).array

    exp.mask.array[20:30, 20:30] = 1

    obs = util.exp2obs(
        exp=exp,
        copy_mask_to=copy_mask_to,
    )
    exp2 = util.obs2exp(
        obs=obs,
        exp=afw_image.ExposureF(exp, deep=True),
        copy_mask_from=copy_mask_to,
    )

    # check images
    assert np.all(obs.image == exp.image.array)
    assert np.all(exp2.image.array == exp.image.array)

    # check weight and variance
    assert np.all(obs.weight == 1/exp.variance.array)
    assert np.all(exp2.variance.array == exp.variance.array)

    # check mask
    if copy_mask_to == 'ormask':
        assert np.all(obs.ormask == exp.mask.array)
    else:
        assert np.all(obs.bmask == exp.mask.array)

    assert np.all(exp2.mask.array == exp.mask.array)

    # check psf
    assert np.all(obs.psf.image == psf_image)

    psf_obj2 = exp2.getPsf()
    psf_image2 = psf_obj2.computeKernelImage(cen).array
    assert np.all(psf_image == psf_image2)
