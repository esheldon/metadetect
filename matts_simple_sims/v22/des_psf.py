import numpy as np
import scipy.interpolate
import galsim


class DESPSF(object):
    def __init__(self, rng, im_width):
        self._rng = rng

        # set the power spectrum and PSF params
        # Heymans et al, 2012 found L0 ~= 3 arcmin, given as 180 arcsec here.
        def _pf(k):
            return (k**2 + (1./180)**2)**(-11./6.)
        self._ps = galsim.PowerSpectrum(
            e_power_function=_pf,
            b_power_function=_pf,
            units='arcsec')
        ng = 64
        gs = max(im_width // ng, 1)
        g1, g2, kappa = self._ps.buildGrid(
            grid_spacing=gs,
            ngrid=ng,
            get_convergence=True,
            variance=0.01**2,
            rng=galsim.BaseDeviate(self._rng.randint(1, 2**30)))
        g1_r, g2_r, mu = galsim.lensing_ps.theoryToObserved(g1, g2, kappa)

        min_gs = (-ng/2 + 0.5) * gs
        max_gs = (ng/2 - 0.5) * gs
        y_gs, x_gs = np.meshgrid(
            np.arange(min_gs, max_gs+gs, gs),
            np.arange(min_gs, max_gs+gs, gs))
        x_gs = x_gs.ravel()
        y_gs = y_gs.ravel()

        self._g1_r = scipy.interpolate.interp2d(x_gs, y_gs, g1_r.ravel())
        self._g2_r = scipy.interpolate.interp2d(x_gs, y_gs, g2_r.ravel())
        self._mu = scipy.interpolate.interp2d(x_gs, y_gs, mu.ravel())

        def _getlogmnsigma(mean, sigma):
            logmean = np.log(mean) - 0.5*np.log(1 + sigma**2/mean**2)
            logvar = np.log(1 + sigma**2/mean**2)
            logsigma = np.sqrt(logvar)
            return logmean, logsigma

        lm, ls = _getlogmnsigma(0.9, 0.1)
        self._fwhm_central = np.exp(self._rng.normal() * ls + lm)

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
        g1 = self._g1_r(x, y)
        g2 = self._g2_r(x, y)
        mu = self._mu(x, y)
        psf = galsim.Moffat(
            beta=2.5,
            fwhm=self._fwhm_central).lens(g1=g1, g2=g2, mu=mu)
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

        return galsim.Convolve(
            self._get_atm(pos.x, pos.y),
            self._get_opt())
