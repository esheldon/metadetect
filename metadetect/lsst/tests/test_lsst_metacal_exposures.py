import numpy as np
from metadetect.lsst.util import exp2obs
from metadetect.lsst.util import get_integer_center, get_jacobian, get_mbexp
from metadetect.lsst.metacal_exposures import (
    get_metacal_exps_fixnoise, get_metacal_mbexps_fixnoise,
)
from copy import deepcopy
import ngmix
from ngmix.metacal import get_all_metacal
import pytest
from lsst.utils import getPackageDir

try:
    getPackageDir('descwl_shear_sims')
    skip_tests_on_simulations = False
except LookupError:
    skip_tests_on_simulations = True


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
def test_metacal_exps(ntrial=10, show=False):
    from descwl_shear_sims.galaxies import make_galaxy_catalog
    from descwl_shear_sims.sim import make_sim
    from descwl_shear_sims.psfs import make_fixed_psf

    seed = None
    dim = 250
    buff = 50

    rng = np.random.RandomState(seed)

    for itrial in range(ntrial):
        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type='fixed',
            layout='grid',
            # gal_config={'mag': 22},
            coadd_dim=dim,
            buff=buff,
        )
        psf = make_fixed_psf(psf_type='gauss')

        sim_data = make_sim(
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=dim,
            se_dim=dim,
            g1=0.02,
            g2=0.00,
            psf=psf,
        )
        exp = sim_data['band_data']['i'][0]
        nexp = deepcopy(exp)
        # copy doesn't copy the psf
        nexp.setPsf(exp.getPsf())
        assert nexp.getWcs() == exp.getWcs()

        obs = exp2obs(exp)
        assert np.all(obs.image == exp.image.array)
        assert np.all(obs.weight == 1/exp.variance.array)
        cen, _ = get_integer_center(exp.getWcs(), exp.getBBox(), as_double=True)

        psf_obj = exp.getPsf()
        psf_image_array = psf_obj.computeKernelImage(cen).array

        assert np.all(obs.psf.image == psf_image_array)

        # gwcs = get_galsim_jacobian_wcs(exp, cen=cen)
        jdata = obs.jacobian._data[0]
        ejac = get_jacobian(exp=exp, cen=cen)
        edata = ejac._data[0]
        for n in jdata.dtype.names:
            assert jdata[n] == edata[n]

        if show:
            compare_images(
                obs.image, exp.image.array, label1='obs', label2='exp',
            )
            compare_images(
                obs.psf.image, psf_image_array,
                label1='obs psf', label2='exp psf',
            )

        noise = np.sqrt(exp.variance.array)
        nexp.image.array[:, :] = noise * rng.normal(size=exp.image.array.shape)
        assert np.all(exp.image.array != nexp.image.array)

        obs.noise = nexp.image.array

        types = ('noshear', '1p', '1m', '2p', '2m')
        mdict_obs = get_all_metacal(
            obs, use_noise_image=True, types=types,
        )
        mdict_exp, noise_mdict = get_metacal_exps_fixnoise(
            exp, nexp, types=types,
        )

        for key in mdict_obs:
            print('checking:', key)
            assert key in mdict_exp, f'checking for {key}'

            timage = mdict_obs[key].image
            eimage = mdict_exp[key].image.array

            tweight = mdict_obs[key].weight
            eweight = 1/mdict_exp[key].variance.array
            if show:
                compare_images(timage, eimage, label1='obs', label2='exp')
            assert np.all(timage == eimage)
            assert np.all(tweight == eweight)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
def test_metacal_mbexp(ntrial=10, show=False):
    from descwl_shear_sims.galaxies import make_galaxy_catalog
    from descwl_shear_sims.sim import make_sim
    from descwl_shear_sims.psfs import make_fixed_psf

    seed = None
    dim = 250
    buff = 50
    bands = ['r', 'i']

    rng = np.random.RandomState(seed)

    for itrial in range(ntrial):
        galaxy_catalog = make_galaxy_catalog(
            rng=rng,
            gal_type='fixed',
            layout='grid',
            # gal_config={'mag': 22},
            coadd_dim=dim,
            buff=buff,
        )
        psf = make_fixed_psf(psf_type='gauss')

        sim_data = make_sim(
            bands=bands,
            rng=rng,
            galaxy_catalog=galaxy_catalog,
            coadd_dim=dim,
            se_dim=dim,
            g1=0.02,
            g2=0.00,
            psf=psf,
        )

        exps = [sim_data['band_data'][band][0] for band in bands]
        noise_exps = []
        mbobs = ngmix.MultiBandObsList()
        for exp in exps:
            nexp = deepcopy(exp)
            nexp.setPsf(exp.getPsf())

            noise = np.sqrt(exp.variance.array)
            nexp.image.array[:, :] = noise * rng.normal(size=exp.image.array.shape)

            noise_exps.append(nexp)

            obs = exp2obs(exp)
            obs.noise = nexp.image.array
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        mbexp = get_mbexp(exps)
        noise_mbexp = get_mbexp(noise_exps)

        types = ('noshear', '1p', '1m', '2p', '2m')
        mdict_obs = get_all_metacal(
            mbobs, use_noise_image=True, types=types,
        )
        mdict_mbexp, noise_mdict = get_metacal_mbexps_fixnoise(
            mbexp, noise_mbexp, types=types,
        )

        for key in mdict_obs:
            print('checking:', key)
            assert key in mdict_mbexp, f'checking for {key}'

            mbobs = mdict_obs[key]
            mbexp = mdict_mbexp[key]
            for iband, band in enumerate(bands):
                print(f'checking band {band}')
                obs = mbobs[iband][0]
                exp = mbexp[band]

                timage = obs.image
                eimage = exp.image.array

                tweight = obs.weight
                eweight = 1/exp.variance.array
                if show:
                    compare_images(timage, eimage, label1='obs', label2='exp')
                assert np.all(timage == eimage)
                assert np.all(tweight == eweight)


def compare_images(im1, im2, label1='im1', label2='im2'):
    import matplotlib.pyplot as mplt
    fig, axs = mplt.subplots(nrows=2, ncols=2)
    axs[1, 1].axis('off')

    axs[0, 0].imshow(im1)
    axs[0, 0].set_title(label1)

    axs[0, 1].imshow(im1)
    axs[0, 1].set_title(label2)

    axs[1, 0].imshow(im1 - im2)
    axs[1, 0].set_title(f'{label1} - {label2}')
    mplt.show()


if __name__ == '__main__':
    # test_metacal_exps(show=False)
    test_metacal_mbexp(show=True)
