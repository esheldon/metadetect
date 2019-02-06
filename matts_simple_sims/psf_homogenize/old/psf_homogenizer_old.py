import numpy as np
import numba
import scipy.interpolate
import scipy.fftpack

import galsim

import ngmix
from ngmix.observation import Observation
from ngmix.fitting import LMCoellip, LMSimple


class FFTPSFHomogenizer(object):
    def __init__(self, psf_model, image_size, grid_spacing,
                 kernel_size=5, dilation_fudge=1.01):
        self.psf_model = psf_model
        self.image_size = image_size
        self.grid_spacing = grid_spacing
        self.kernel_size = kernel_size
        self.dilation_fudge = dilation_fudge

    def apply_kernel(self, image):
        return convolve_variable_kernel(self.get_kernel, image)

    def get_kernel(self, row, col):
        _row = min(row, self.image_size[0]-1 - 1e-3)
        _col = min(col, self.image_size[1]-1 - 1e-3)
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
        ni = self.image_size[0] // self.grid_spacing
        nj = self.image_size[1] // self.grid_spacing

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
        ftgt = scipy.fftpack.fft2(self._target_psf.image.copy())

        self._kernel_imgs = {}
        for k, psf_obs in self._psf_obs.items():
            fpsf = scipy.fftpack.fft2(psf_obs.image.copy())

            krn = scipy.fftpack.fftshift(scipy.fftpack.ifft2(ftgt/fpsf).real)
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
            gim = gim.dilate(self.dilation_fudge)
            target_psf = gim.drawImage(
                nx=self._psf_im_shape[1],
                ny=self._psf_im_shape[0],
                scale=1).array

        target_psf /= np.sum(target_psf)
        self._target_psf = Observation(target_psf, jacobian=self._jacob)

    def _fit_all_psfs(self):
        n_gauss = 2
        self._psf_gmix = {}
        self._psf_obs = {}
        ni = self.image_size[0] // self.grid_spacing
        nj = self.image_size[1] // self.grid_spacing
        self._psf_im_shape = None
        for i in range(ni+1):
            row = min(i * self.grid_spacing, self.image_size[0]-1)
            for j in range(nj+1):
                col = min(j * self.grid_spacing, self.image_size[1]-1)

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


class NGMixPSFHomogenizer(object):
    def __init__(self, psf_model, image_size, grid_spacing,
                 kernel_size=5, n_gauss=2,
                 kernel_n_gauss=1, dilation_fudge=1.01):
        self.psf_model = psf_model
        self.image_size = image_size
        self.grid_spacing = grid_spacing
        self.kernel_size = kernel_size
        self.n_gauss = n_gauss
        self.kernel_n_gauss = kernel_n_gauss
        self.dilation_fudge = dilation_fudge

    def apply_kernel(self, image):
        return convolve_variable_kernel(self.get_kernel, image)

    def get_kernel(self, row, col):
        kc = (self.kernel_size - 1)/2
        kdim = [self.kernel_size] * 2
        kjac = ngmix.UnitJacobian(row=kc, col=kc)

        gmix = self._get_kernel_gmix(row, col)
        kim = gmix.make_image(kdim, jacobian=kjac)
        return kim / np.sum(kim)

    def _get_kernel_gmix(self, row, col):
        _row = min(row, self.image_size[0]-1 - 1e-3)
        _col = min(col, self.image_size[1]-1 - 1e-3)
        parr = np.zeros(self._n_pars)
        for pind, interp in self._interps.items():
            parr[pind] = interp(_col, _row)
        return ngmix.GMixCoellip(pars=parr)

    def solve_for_kernel(self):
        self._fit_all_psfs()
        self._set_target_psf()
        self._fit_kernels()
        self._interp_kernels()

    def _interp_kernels(self):
        n_pars = len(self._kernel_pars[(0, 0)])
        self._n_pars = n_pars
        ni = self.image_size[0] // self.grid_spacing
        nj = self.image_size[1] // self.grid_spacing

        self._interps = {}
        for pind in range(n_pars):
            parr = np.zeros((ni+1, nj+1))
            for k, v in self._kernel_pars.items():
                ki = k[0] // self.grid_spacing
                kj = k[1] // self.grid_spacing
                parr[ki, kj] = v[pind]
            self._interps[pind] = scipy.interpolate.interp2d(
                np.arange(ni+1) * self.grid_spacing,
                np.arange(nj+1) * self.grid_spacing,
                parr,
                kind='linear')

    def _fit_kernels(self):
        self._kernel_gmix = {}
        self._kernel_pars = {}
        for k, psf_obs in self._psf_obs.items():
            tgt = self._target_psf.copy()
            tgt.set_psf(psf_obs)

            if False:
                cen_prior = ngmix.priors.CenPrior(0.0, 0.0, 1.0, 1.0)
                g_prior = ngmix.priors.GPriorBA(0.2)
                T_prior = ngmix.priors.FlatPrior(1e-6, 1e6)
                F_prior = ngmix.priors.FlatPrior(1e-6, 1e6)

                prior = ngmix.joint_prior.PriorCoellipSame(
                    self.kernel_n_gauss,
                    cen_prior,
                    g_prior,
                    T_prior,
                    F_prior)

                fitter = LMCoellip(tgt, self.kernel_n_gauss, prior=prior)
                guess = np.zeros(4 + self.kernel_n_gauss * 2)
                guess[4:] += 1

                fitter.go(guess)
                res = fitter.get_result()
                self._kernel_pars[k] = res['pars']
                kern_gmix = ngmix.GMixCoellip(pars=res['pars'])
                self._kernel_gmix[k] = kern_gmix
            else:
                fitter = LMSimple(tgt, "gauss")
                guess = np.zeros(6) + 1e-3
                guess[4:] += 1
                fitter.go(guess)
                res = fitter.get_result()
                self._kernel_pars[k] = res['pars']
                kern_gmix = ngmix.GMixModel(res['pars'], "gauss")
                self._kernel_gmix[k] = kern_gmix

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
            gim = gim.dilate(self.dilation_fudge)
            target_psf = gim.drawImage(
                nx=self._psf_im_shape[1],
                ny=self._psf_im_shape[0],
                scale=1).array

        target_psf /= np.sum(target_psf)
        self._target_psf = Observation(target_psf, jacobian=self._jacob)

    def _fit_all_psfs(self):
        self._psf_gmix = {}
        self._psf_obs = {}
        ni = self.image_size[0] // self.grid_spacing
        nj = self.image_size[1] // self.grid_spacing
        self._psf_im_shape = None
        for i in range(ni+1):
            row = min(i * self.grid_spacing, self.image_size[0]-1)
            for j in range(nj+1):
                col = min(j * self.grid_spacing, self.image_size[1]-1)

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

                pfitter = LMCoellip(psf_obs, self.n_gauss)
                guess = np.zeros(4 + self.n_gauss*2) + 1e-3
                guess[4:] = guess[4:] + 1
                guess[-self.n_gauss] = (
                    guess[-self.n_gauss] / np.sum(guess[-self.n_gauss]))
                pfitter.go(guess)
                psf_gmix = pfitter.get_gmix()

                self._psf_gmix[(row, col)] = psf_gmix

                psf_obs.set_gmix(psf_gmix)
                self._psf_obs[(row, col)] = psf_obs


@numba.jit([
    "float64[:,:](pyobject,float64[:,:])",
    "float32[:,:](pyobject,float32[:,:])"])
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
    # TODO:
    # 1. switch to an inplace algorithm
    # 2. do the rows in batches to keep the cache more coherent
    # 3. maybe add a type hint to help numba generate better code
    im_new = np.zeros_like(image)
    i_shape = image.shape[0]
    j_shape = image.shape[1]
    nk = kernel(0, 0).shape[0]
    dk = (nk-1) // 2
    for i_new in range(i_shape):
        for j_new in range(j_shape):
            k = kernel(i_new, j_new)

            i_old_start = max(0, i_new-dk)
            j_old_start = max(0, j_new-dk)

            i_old_end = min(i_new + dk + 1, i_shape)
            j_old_end = min(j_new + dk + 1, j_shape)

            i_kern_start = max(0, dk-i_new)
            j_kern_start = max(0, dk-j_new)

            n_i_kern = i_old_end - i_old_start
            n_j_kern = j_old_end - j_old_start

            _sum = 0.0
            for _i in range(n_i_kern):
                for _j in range(n_j_kern):
                    _sum += (
                        image[i_old_start + _i, j_old_start + _j] *
                        k[i_kern_start + _i, j_kern_start + _j])
            im_new[i_new, j_new] = _sum
    return im_new
