import numpy as np
import logging
import ngmix
from ngmix.gexceptions import BootPSFFailure
import esutil as eu
from .util import Namer
from . import procflags

logger = logging.getLogger(__name__)


class FitterBase(dict):
    """
    we don't create a new instance of this for each fit, because
    the prior can be set once
    """
    def __init__(self, config, rng):

        self.rng = rng
        self.update(config)

    def go(self, mbobs_list):
        """
        do measurements.
        """
        raise NotImplementedError("implement go()")


class Moments(FitterBase):
    """
    measure simple weighted moments
    """
    def __init__(self, *args, **kw):
        super(Moments, self).__init__(*args, **kw)
        self._set_mompars()

    def go(self, mbobs_list):
        """
        run moments measurements on all objects

        parameters
        ----------
        mbobs_list: list of ngmix.MultiBandObsList
            One for each object to be measured

        returns
        -------
        output: list of numpy arrays with fields
            Results for each object
        """

        datalist = []
        for i, mbobs in enumerate(mbobs_list):

            if not self._check_flags(mbobs):
                res = {
                    'flags': procflags.IMAGE_FLAGS,
                    'flagstr': procflags.get_name(procflags.IMAGE_FLAGS),
                }
                pres = {
                    'flags': procflags.NO_ATTEMPT,
                    'flagstr': procflags.get_name(procflags.NO_ATTEMPT),
                }
            else:

                obs = self._do_coadd_maybe(mbobs)

                pres = self._measure_moments(obs.psf)
                res = self._measure_moments(obs)

            if res['flags'] != 0:
                logger.debug("        moments failed: %s" % res['flagstr'])

            if pres['flags'] != 0:
                logger.debug('        psf moments '
                             'failed: %s' % pres['flagstr'])

            fit_data = self._get_output(res, pres)

            if res['flags'] == 0 and pres['flags'] == 0:
                self._print_result(fit_data)

            datalist.append(fit_data)

        if len(datalist) == 0:
            return None
        else:
            return eu.numpy_util.combine_arrlist(datalist)

    def _get_max_psf_size(self, mbobs):
        sizes = []
        for obslist in mbobs:
            for obs in obslist:
                sizes.append(obs.psf.image.shape[0])
        return max(sizes)

    def _maybe_zero_pad_image(self, im, size):
        if im.shape[0] == size:
            return im
        elif im.shape[0] < size:
            diff = size - im.shape[0]
            assert diff % 2 == 0, "Can only pad images with even padding!"
            half_diff = diff // 2
            new_im = np.zeros((size, size), dtype=im.dtype)
            new_im[half_diff:-half_diff, half_diff:-half_diff] = im

            newcen = (size - 1) // 2
            oldcen = (im.shape[0] - 1) // 2
            assert new_im[newcen, newcen] == im[oldcen, oldcen]
            return new_im
        else:
            raise ValueError("cannot pad image to a smaller size!")

    def _do_coadd_maybe(self, mbobs):
        """
        coadd all images and psfs.  Assume perfect registration and
        same wcs
        """

        # note here assuming we can re-use the wcs etc.
        new_obs = mbobs[0][0].copy()

        if len(mbobs) == 1 and len(mbobs[0]) == 1:
            return new_obs

        max_psf_size = self._get_max_psf_size(mbobs)

        first = True
        wsum = 0.0
        for obslist in mbobs:
            for obs in obslist:
                tim = obs.image
                twt = obs.weight
                tpsf_im = self._maybe_zero_pad_image(
                    obs.psf.image, max_psf_size)
                tpsf_wt = obs.psf.weight

                medweight = np.median(twt)
                noise = np.sqrt(1.0/medweight)

                psf_medweight = np.median(tpsf_wt)
                psf_noise = np.sqrt(1.0/psf_medweight)

                tnim = self.rng.normal(size=tim.shape, scale=noise)
                tpsf_nim = self.rng.normal(size=tpsf_im.shape, scale=psf_noise)

                wsum += medweight

                if first:
                    im = tim*medweight
                    psf_im = tpsf_im*medweight

                    nim = tnim * medweight
                    psf_nim = tpsf_nim * medweight

                    first = False
                else:
                    im += tim*medweight
                    psf_im += tpsf_im*medweight

                    nim += tnim * medweight
                    psf_nim += tpsf_nim * medweight

        fac = 1.0/wsum
        im *= fac
        psf_im *= fac

        nim *= fac
        psf_nim *= fac

        noise_var = nim.var()
        psf_noise_var = psf_nim.var()

        wt = np.zeros(im.shape) + 1.0/noise_var
        psf_wt = np.zeros(psf_im.shape) + 1.0/psf_noise_var

        new_obs.set_image(im, update_pixels=False)
        new_obs.set_weight(wt)

        new_obs.psf.set_image(psf_im, update_pixels=False)
        new_obs.psf.set_weight(psf_wt)

        return new_obs

    def _print_result(self, data):
        mess = "        wmom s2n: %g Trat: %g"
        logger.debug(mess % (data['wmom_s2n'][0], data['wmom_T_ratio'][0]))

    def _measure_moments(self, obs):
        """
        measure weighted moments
        """

        res = self.weight.get_weighted_moments(obs=obs, maxrad=1.e9)

        if res['flags'] != 0:
            return res

        res['numiter'] = 1
        res['g'] = res['e']
        res['g_cov'] = res['e_cov']

        return res

    def _get_dtype(self, model, npars):
        n = Namer(front=model)
        dt = [
            ('flags', 'i4'),

            ('psfrec_flags', 'i4'),  # psfrec is the original psf
            ('psfrec_g', 'f8', 2),
            ('psfrec_T', 'f8'),

            ('psf_flags', 'i4'),
            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),

            (n('flags'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('g'), 'f8', 2),
            (n('g_cov'), 'f8', (2, 2)),
            (n('T'), 'f8'),
            (n('T_err'), 'f8'),
            (n('T_ratio'), 'f8'),
        ]

        return dt

    def _get_output(self, res, pres):

        npars = 6

        model = 'wmom'
        n = Namer(front=model)

        dt = self._get_dtype(model, npars)
        output = np.zeros(1, dtype=dt)

        output['psfrec_flags'] = procflags.NO_ATTEMPT

        output['psf_flags'] = pres['flags']
        output[n('flags')] = res['flags']

        flags = 0
        if pres['flags'] != 0:
            flags |= procflags.PSF_FAILURE

        if res['flags'] != 0:
            flags |= procflags.OBJ_FAILURE

        if pres['flags'] == 0:
            output['psf_g'] = pres['g']
            output['psf_T'] = pres['T']

        if res['flags'] == 0:
            output[n('s2n')] = res['s2n']
            output[n('pars')] = res['pars']
            output[n('g')] = res['g']
            output[n('g_cov')] = res['g_cov']
            output[n('T')] = res['T']
            output[n('T_err')] = res['T_err']

            if pres['flags'] == 0:
                output[n('T_ratio')] = res['T']/pres['T']

        output['flags'] = flags
        return output

    def _set_mompars(self):
        wpars = self['weight']

        T = ngmix.moments.fwhm_to_T(wpars['fwhm'])

        # the weight is always centered at 0, 0 or the
        # center of the coordinate system as defined
        # by the jacobian

        weight = ngmix.GMixModel(
            [0.0, 0.0, 0.0, 0.0, T, 1.0],
            'gauss',
        )

        # make the max of the weight 1.0 to get better
        # fluxes

        weight.set_norms()
        norm = weight.get_data()['norm'][0]
        weight.set_flux(1.0/norm)

        self.weight = weight

    def _check_flags(self, mbobs):
        """
        only one image per band, no epochs, so anything that hits an edge
        """
        flags = self['bmask_flags']

        isok = True
        if flags is not None:
            for obslist in mbobs:
                if len(obslist) == 0:
                    isok = False
                    break

                for obs in obslist:
                    w = np.where((obs.bmask & flags) != 0)
                    if w[0].size > 0:
                        logger.info("   EDGE HIT")
                        isok = False
                        break

        return isok


def fit_all_psfs(mbobs, psf_conf, rng):
    """
    fit all psfs in the input observations
    """

    for obslist in mbobs:
        for obs in obslist:
            psf_obs = obs.get_psf()
            fit_one_psf(psf_obs, psf_conf, rng)


def fit_one_psf(obs, pconf, rng):
    # Tguess=4.0*obs.jacobian.get_scale()**2
    fwhm_guess = 0.9
    Tguess = ngmix.moments.fwhm_to_T(fwhm_guess)

    if 'coellip' in pconf['model']:
        ngauss = ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner = ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
            rng=rng,
        )

    else:
        runner = ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
            rng=rng,
        )

    runner.go(ntry=pconf['ntry'])

    psf_fitter = runner.fitter
    res = psf_fitter.get_result()
    obs.update_meta_data({'fitter': psf_fitter})

    if res['flags'] == 0:
        gmix = psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))
