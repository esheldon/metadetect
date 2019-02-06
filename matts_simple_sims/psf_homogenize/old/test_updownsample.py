import galsim
import numpy as np
import pytest

from psf_homogenizer import upsample_image, downsample_image

SCALE = 0.25


def test_upsample_image():
    gauss = galsim.Gaussian(fwhm=1).drawImage(nx=11, ny=13, scale=0.25).array

    gauss_ssamp = upsample_image(gauss, 4, ensure_odd=True)
    assert gauss_ssamp.shape[0] == 53
    assert gauss_ssamp.shape[1] == 45

    cen = (
        (gauss_ssamp.shape[0] - 1) // 2,
        (gauss_ssamp.shape[1] - 1) // 2)

    assert np.sum(gauss_ssamp == gauss_ssamp[cen[0], cen[1]]) == 1

    gauss_ssamp = upsample_image(gauss, 4, ensure_odd=False)
    assert gauss_ssamp.shape[0] == 52
    assert gauss_ssamp.shape[1] == 44


def test_downsample_image():
    gauss = galsim.Gaussian(fwhm=1).drawImage(nx=46, ny=58, scale=0.25).array

    new_gauss = downsample_image(gauss, 4, (11, 13))
    assert new_gauss.shape == (11, 13)

    cen = (
        (new_gauss.shape[0] - 1) // 2,
        (new_gauss.shape[1] - 1) // 2)

    assert np.sum(new_gauss == new_gauss[cen[0], cen[1]]) == 1


@pytest.mark.parametrize('fac', [1, 2, 2.5, 4])
def test_roundtrip(fac):
    gauss = galsim.Gaussian(fwhm=1).drawImage(nx=11, ny=13, scale=0.25).array

    gauss_usamp = upsample_image(gauss, fac, ensure_odd=True)
    gauss_usamp_dsamp = downsample_image(gauss_usamp, fac, [13, 11])

    assert np.allclose(gauss_usamp_dsamp, gauss, atol=0, rtol=1e-1)
