import numpy as np
import numba
import scipy.interpolate
import scipy.fftpack

import galsim

import ngmix
from ngmix.observation import Observation
from ngmix.fitting import LMCoellip


class PSFHomogenizer(object):
    def __init__(self, psf_model, image_size, grid_spacing,
                 kernel_size=5, dilation_fudge=1.025):
        self.psf_model = psf_model
        self.image_size = image_size
        self.grid_spacing = grid_spacing
        self.kernel_size = kernel_size
        self.dilation_fudge = dilation_fudge

    def apply_kernel(self, image):
        return convolve_variable_kernel(self.get_kernel, image)

    def get_kernel(self, row, col):
        _row = min(row, self.image_size-1 - 1e-3)
        _col = min(col, self.image_size-1 - 1e-3)
        parr = np.zeros(self._n_pars)
        for pind, interp in self._interps.items():
            parr[pind] = interp(_col, _row)
        return parr.reshape(self.kernel_size, self.kernel_size)

    def solve_for_kernel(self):
        self._fit_all_psfs()
        self._set_target_psf()
        self._fit_kernels()
        self._interp_kernels()

    def _interp_kernels(self):
        n_pars = len(self._kernel_imgs[(0, 0)].ravel())
        self._n_pars = n_pars
        ni = self.image_size // self.grid_spacing
        nj = self.image_size // self.grid_spacing

        self._interps = {}
        for pind in range(n_pars):
            parr = np.zeros((ni+1, nj+1))
            for k, v in self._kernel_imgs.items():
                ki = k[0] // self.grid_spacing
                kj = k[1] // self.grid_spacing
                parr[ki, kj] = v.ravel()[pind]
            self._interps[pind] = scipy.interpolate.interp2d(
                np.arange(ni+1) * self.grid_spacing,
                np.arange(nj+1) * self.grid_spacing,
                parr,
                kind='linear')

    def _fit_kernels(self):
        sz = (self._psf_im_shape[0] - self.kernel_size) // 2
        ftgt = scipy.fftpack.fft2(
            np.pad(
                self._target_psf.image,
                self._target_psf.image.shape[0] * 2,
                'constant', constant_values=0.0))

        self._kernel_imgs = {}
        for k, psf_obs in self._psf_obs.items():
            fpsf = scipy.fftpack.fft2(
                np.pad(
                    psf_obs.image,
                    psf_obs.image.shape[0] * 2,
                    'constant', constant_values=0.0))

            krn = scipy.fftpack.fftshift(scipy.fftpack.ifft2(ftgt/fpsf).real)
            sz = (krn.shape[0] - self.kernel_size) // 2
            krn = krn[sz:-sz, sz:-sz]
            krn = krn / np.sum(krn)
            self._kernel_imgs[k] = krn

    def _set_target_psf(self):
        # find the largets kernel according to
        #  sqrt(T) * dilation
        # where
        # g = sqrt(shear.g1**2 + shear.g2**2)
        # dilation = 1.0 + 2.0*g
        big_key = None
        big_fac = None
        for k, v in self._psf_gmix.items():
            g1, g2, T = v.get_g1g2T()
            gfac = 1.0 + 2.0 * np.sqrt(g1*g1 + g2*g2)
            fac = np.sqrt(T) * gfac
            if big_fac is None or fac > big_fac:
                big_key = k
                big_fac = fac

        target_psf = self.psf_model(*big_key)
        assert self._psf_im_shape == target_psf.shape

        # dilate the target PSF if requested
        if self.dilation_fudge > 1:
            gim = galsim.InterpolatedImage(galsim.ImageD(target_psf), scale=1)
            fwhm = np.sqrt(self.dilation_fudge**2 - 1) * gim.calculateFWHM()
            gim = galsim.Convolve(gim, galsim.Gaussian(fwhm=fwhm))
            target_psf = gim.drawImage(
                nx=self._psf_im_shape[1],
                ny=self._psf_im_shape[0],
                scale=1,
                method='no_pixel').array

        target_psf /= np.sum(target_psf)
        self._target_psf = Observation(target_psf, jacobian=self._jacob)

    def _fit_all_psfs(self):
        n_gauss = 2
        self._psf_gmix = {}
        self._psf_obs = {}
        ni = self.image_size // self.grid_spacing
        nj = self.image_size // self.grid_spacing
        self._psf_im_shape = None
        for i in range(ni+1):
            row = min(i * self.grid_spacing, self.image_size-1)
            for j in range(nj+1):
                col = min(j * self.grid_spacing, self.image_size-1)

                psf_im = self.psf_model(row+1, col+1)
                psf_im /= np.sum(psf_im)
                if self._psf_im_shape is None:
                    self._psf_im_shape = psf_im.shape
                    self._jacob = ngmix.UnitJacobian(
                        row=(self._psf_im_shape[0] - 1)/2,
                        col=(self._psf_im_shape[1] - 1)/2)
                else:
                    assert self._psf_im_shape == psf_im.shape

                psf_obs = Observation(psf_im, jacobian=self._jacob)

                pfitter = LMCoellip(psf_obs, n_gauss)
                guess = np.zeros(4 + n_gauss*2) + 1e-3
                guess[4:] = guess[4:] + 1
                guess[-n_gauss] = (
                    guess[-n_gauss] / np.sum(guess[-n_gauss]))
                pfitter.go(guess)
                psf_gmix = pfitter.get_gmix()

                self._psf_gmix[(row, col)] = psf_gmix

                psf_obs.set_gmix(psf_gmix)
                self._psf_obs[(row, col)] = psf_obs


def upsample_image(
        image, upsampling_factor, ensure_odd=False, kind='lanczos7'):
    im = galsim.InterpolatedImage(
        galsim.ImageD(image, wcs=galsim.PixelScale(1)),
        x_interpolant=kind)

    new_x = int(image.shape[1] * upsampling_factor)
    new_y = int(image.shape[0] * upsampling_factor)
    if ensure_odd:
        if new_x % 2 == 0:
            new_x += 1
        if new_y % 2 == 0:
            new_y += 1
    im_out = im.drawImage(
        nx=new_x, ny=new_y, scale=1.0/upsampling_factor, method='no_pixel')
    return im_out.array


def downsample_image(
        image, downsampling_factor, target_size, kind='lanczos7'):
    im = galsim.InterpolatedImage(
        galsim.ImageD(image, wcs=galsim.PixelScale(1)),
        x_interpolant=kind)
    im_out = im.drawImage(
        nx=target_size[1], ny=target_size[0], scale=downsampling_factor,
        method='no_pixel')
    return im_out.array


@numba.jit
def convolve_variable_kernel(kernel, image):
    """Convolve a variable kernel into an image.

    Parameters
    ----------
    kernel : callable
        A function that takes in the location in the image
        in the form of a zero-indexed (row, col) and outputs the
        kernel for that location in the image. The function signature should
        be `kernel(row, col)`.
    image : array-like, two-dimensional
        The image with which to convolve the kernel.

    Returns
    -------
    conv : np.ndarray, two-dimensional
        The convolved image.
    """
    # TODO: switch to an inplace algorithm?
    im_new = np.zeros_like(image)
    i_shape = image.shape[0]
    j_shape = image.shape[1]
    nk = kernel(0, 0).shape[0]
    dk = (nk-1) // 2

    # we do the convolution in batches to avoid having to mix python and
    # efficient(?) numba code
    batch_size = nk * 2
    n_batches = i_shape // batch_size
    if n_batches * nk < i_shape:
        n_batches += 1

    kernels = np.zeros((batch_size, j_shape, nk, nk), dtype=kernel(0, 0).dtype)

    for batch in range(n_batches):
        batch_start = batch * batch_size
        batch_end = min(batch_start + batch_size, i_shape)

        # 1. get all of the kernels
        for i_new in range(batch_start, batch_end):
            for j_new in range(j_shape):
                kernels[i_new - batch_start, j_new, :, :] \
                    = kernel(i_new, j_new)

        # 2. do the convolution
        _convolve_variable_kernel_batch(
            batch_start, batch_end,
            i_shape, j_shape, dk, kernels, im_new, image)

    return im_new


@numba.njit
def _convolve_variable_kernel_batch(
        batch_start, batch_end, i_shape, j_shape, dk, kernels, im_new, image):
    for i_new in range(batch_start, batch_end):
        i_old_start = np.int64(np.maximum(0, i_new-dk))
        i_old_end = np.int64(np.minimum(i_new + dk + 1, i_shape))
        i_kern_start = np.int64(np.maximum(0, dk-i_new))
        n_i_kern = i_old_end - i_old_start

        for j_new in range(j_shape):
            j_old_start = np.int64(np.maximum(0, j_new-dk))
            j_old_end = np.int64(np.minimum(j_new + dk + 1, j_shape))
            j_kern_start = np.int64(np.maximum(0, dk-j_new))
            n_j_kern = j_old_end - j_old_start

            _sum = 0.0
            for _i in range(n_i_kern):
                for _j in range(n_j_kern):
                    _sum += (
                        image[i_old_start + _i, j_old_start + _j] *
                        kernels[i_new - batch_start, j_new,
                                i_kern_start + _i, j_kern_start + _j])
            im_new[i_new, j_new] = _sum
