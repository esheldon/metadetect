import numpy as np
import galsim


class DESPSF(object):
    def __init__(self, rng, im_width, scale, trunc=10):
        self._rng = rng
        self._x_scale = 2.0 / im_width / scale
        self._im_cen = (im_width - 1)/2
        self._scale = scale

        # set the power spectrum and PSF params
        # Heymans et al, 2012 found L0 ~= 3 arcmin, given as 180 arcsec here.
        def _pf(k):
            return (k**2 + (1./180)**2)**(-11./6.) * np.exp(-(k*trunc)**2)
        self._ps = galsim.PowerSpectrum(
            e_power_function=_pf,
            b_power_function=_pf)
        ng = 64
        gs = max(im_width * self._scale / ng, 1)
        self._ps.buildGrid(
            grid_spacing=gs,
            ngrid=ng,
            get_convergence=True,
            variance=0.01**2,
            rng=galsim.BaseDeviate(self._rng.randint(1, 2**30)))

        def _getlogmnsigma(mean, sigma):
            logmean = np.log(mean) - 0.5*np.log(1 + sigma**2/mean**2)
            logvar = np.log(1 + sigma**2/mean**2)
            logsigma = np.sqrt(logvar)
            return logmean, logsigma

        lm, ls = _getlogmnsigma(0.9, 0.1)
        self._fwhm_central = np.exp(self._rng.normal() * ls + lm)

        ls = 0.001
        lm = 0.02 / 5 * 10 / trunc
        self._fwhm_x = self._rng.normal() * ls + lm
        self._fwhm_y = self._rng.normal() * ls + lm
        self._fwhm_xx = self._rng.normal() * ls + lm / 10
        self._fwhm_xy = self._rng.normal() * ls + lm / 10
        self._fwhm_yy = self._rng.normal() * ls + lm / 10

        # these are all properly normalized to an RMS abberation of 0.26
        # >>> vals = np.array(
        #    [ 0.13, 0.13, 0.14, 0.06, 0.06, 0.05, 0.06, 0.03])
        # >>> 0.26 * vals / np.sqrt(np.sum(vals**2))
        # array([0.13, 0.13, 0.14, 0.06, 0.06, 0.05, 0.06, 0.03])
        self._abbers = dict(
            defocus=self._rng.normal() * 0.13,
            astig1=self._rng.normal() * 0.13,
            astig2=self._rng.normal() * 0.14,
            coma1=self._rng.normal() * 0.06,
            coma2=self._rng.normal() * 0.06,
            trefoil1=self._rng.normal() * 0.05,
            trefoil2=self._rng.normal() * 0.06,
            spher=self._rng.normal() * 0.03)

    def _get_atm(self, x, y):
        xs = (x - 1 - self._im_cen) * self._x_scale
        ys = (y - 1 - self._im_cen) * self._x_scale
        g1, g2 = self._ps.getShear((xs, ys))
        mu = self._ps.getMagnification((xs, ys))
        fwhm = (
            self._fwhm_central +
            xs * self._fwhm_x +
            ys * self._fwhm_y +
            xs * xs * self._fwhm_xx +
            xs * ys * self._fwhm_xy +
            ys * ys * self._fwhm_yy)
        psf = galsim.Moffat(
            beta=2.5,
            fwhm=fwhm).lens(g1=g1, g2=g2, mu=mu)
        return psf

    def _get_opt(self):
        psf = galsim.OpticalPSF(
            lam_over_diam=800 * 1.e-9 / 4 * 206265,  # arcsec
            obscuration=0.3,
            nstruts=4,
            strut_thick=0.05,
            strut_angle=10*galsim.degrees,
            **self._abbers)

        return psf

    def getPSF(self, pos):
        return self._get_atm(pos.x, pos.y)
