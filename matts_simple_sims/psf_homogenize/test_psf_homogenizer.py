import galsim
import numpy as np

from psf_homogenizer import PSFHomogenizer

SCALE = 0.25
PSF_FAC = 0.95
GFWHM = 1e-6


def test_gaussian_smoke():
    obj = [
        galsim.Gaussian(fwhm=GFWHM).shift(-12 - SCALE/2, SCALE/2),
        galsim.Gaussian(fwhm=GFWHM).shift(12 + SCALE/2, SCALE/2)]

    pobj = [
        galsim.Convolve(obj[0], galsim.Gaussian(fwhm=PSF_FAC)),
        galsim.Convolve(obj[1], galsim.Gaussian(fwhm=1.00))]

    def psf_model(row, col):
        if col >= 160/2:
            return galsim.Gaussian(fwhm=1.0).drawImage(
                scale=SCALE, nx=33, ny=33).array
        else:
            return galsim.Gaussian(fwhm=PSF_FAC).drawImage(
                scale=SCALE, nx=33, ny=33).array

    hpsf = PSFHomogenizer(psf_model, [160, 160],)

    im_orig = galsim.Sum(pobj).drawImage(scale=SCALE, nx=160, ny=160).array
    imh = hpsf.homogenize_image(im_orig)

    def _subtract_mirrored(im, im_true):
        imn = np.zeros_like(im)
        mid = imn.shape[1] // 2
        imn[:, :mid] = im[:, :mid] - im_true[:, mid:][:, ::-1]
        imn[:, mid:] = im[:, mid:] - im_true[:, mid:]
        return imn

    max_diff_hmg = np.max(np.abs(_subtract_mirrored(imh, imh)/imh.max()))
    max_diff_orig = np.max(np.abs(
        _subtract_mirrored(im_orig, im_orig)/im_orig.max()))

    assert max_diff_orig > 1e-2
    assert max_diff_hmg < max_diff_orig / 10
