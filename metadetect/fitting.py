import numpy as np
import logging
import ngmix
from ngmix.gexceptions import BootPSFFailure, BootGalFailure
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

    def _get_dtype(self, model, npars, flux_nband=1):
        n = Namer(front=model)
        dt = [
            ('flags', 'i4'),

            ('psfrec_flags', 'i4'),  # psfrec is the original psf
            ('psfrec_g', 'f8', 2),
            ('psfrec_T', 'f8'),

            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),

            # raw mom is always Mf, Mr, Mp, Mc
            # in ngmix this is flux, T, M1, M2
            # e.g., e1 = Mp/T = M1/T and e2 = Mc/T = M2/T
            # these moments are not normalized by the total flux
            (n('raw_mom'), 'f8', 4),
            (n('raw_mom_cov'), 'f8', (4, 4)),

            (n('flags'), 'i4'),
            (n('s2n'), 'f8'),
            (n('pars'), 'f8', npars),
            (n('g'), 'f8', 2),
            (n('g_cov'), 'f8', (2, 2)),
            (n('T'), 'f8'),
            (n('T_err'), 'f8'),
            (n('T_ratio'), 'f8'),
        ]
        if flux_nband > 1:
            dt += [
                (n('flux'), 'f8', flux_nband),
                (n('flux_err'), 'f8', flux_nband),
            ]
        else:
            dt += [
                (n('flux'), 'f8'),
                (n('flux_err'), 'f8'),
            ]

        return dt

    def _get_output(self, res, pres):

        npars = 6

        model = 'wmom'
        n = Namer(front=model)

        dt = self._get_dtype(model, npars)
        output = np.zeros(1, dtype=dt)

        output['psfrec_flags'] = procflags.NO_ATTEMPT

        output[n('flags')] = res['flags']

        flags = 0
        if pres['flags'] != 0:
            flags |= procflags.PSF_FAILURE

        if res['flags'] != 0:
            flags |= procflags.OBJ_FAILURE

        if pres['flags'] == 0:
            output['psf_g'] = pres['g']
            output['psf_T'] = pres['T']

        if 'sums' in res and 'sums_cov' in res:
            # we always keep the raw moments as long as they are there
            # 5, 4, 2, 3 is the magic indexing from the ngmix moments code
            output[n('raw_mom')] = np.array([
                res['sums'][5],
                res['sums'][4],
                res['sums'][2],
                res['sums'][3],
            ])
            for inew, iold in enumerate([5, 4, 2, 3]):
                for jnew, jold in enumerate([5, 4, 2, 3]):
                    output[n('raw_mom_cov')][0, inew, jnew] \
                        = res['sums_cov'][iold, jold]
        else:
            flags |= procflags.NOMOMENTS_FAILURE

        if res['flags'] == 0:
            output[n('s2n')] = res['s2n']
            output[n('pars')] = res['pars']
            output[n('g')] = res['g']
            output[n('g_cov')] = res['g_cov']
            output[n('T')] = res['T']
            output[n('T_err')] = res['T_err']
            output[n('flux')] = res['flux']
            output[n('flux_err')] = res['flux_err']

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


class KSigmaMoments(Moments):
    """
    measure pre-PSF 'ksigma' weighted moments following Bernstein et al. but
    in real-space.
    """
    def __init__(self, *args, **kw):
        super(Moments, self).__init__(*args, **kw)
        self._set_fitter()

    def _print_result(self, data):
        mess = "        ksigma s2n: %g Trat: %g"
        logger.debug(mess % (data['ksigma_s2n'][0], data['ksigma_T_ratio'][0]))

    def _measure_moments(self, obs):
        """
        measure weighted moments
        """

        if obs.has_psf():
            res = self._fitter.go(obs)
        else:
            res = self._fitter.go(obs, no_psf=True)

        res['numiter'] = 1
        res['g'] = res['e']
        res['g_cov'] = res['e_cov']
        res['g_err'] = res['e_err']

        return res

    def _get_output(self, res, pres):

        npars = 6

        model = 'ksigma'
        n = Namer(front=model)

        dt = self._get_dtype(model, npars)
        output = np.zeros(1, dtype=dt)

        output['psfrec_flags'] = procflags.NO_ATTEMPT

        output[n('flags')] = res['flags']

        flags = 0
        if pres['flags'] != 0:
            flags |= procflags.PSF_FAILURE

        if res['flags'] != 0:
            flags |= procflags.OBJ_FAILURE

        if pres['flags'] == 0:
            output['psf_g'] = pres['g']
            output['psf_T'] = pres['T']

        if 'mom' in res and 'mom_cov' in res:
            # we always keep the raw moments as long as they are there
            # 5, 4, 2, 3 is the magic indexing from the ngmix moments code
            output[n('raw_mom')] = res['mom']
            output[n('raw_mom_cov')][:] = res['mom_cov']
        else:
            flags |= procflags.NOMOMENTS_FAILURE

        if res['flags'] == 0:
            output[n('s2n')] = res['s2n']
            output[n('pars')] = res['pars']
            output[n('g')] = res['g']
            output[n('g_cov')] = res['g_cov']
            output[n('T')] = res['T']
            output[n('T_err')] = res['T_err']
            output[n('flux')] = res['flux']
            output[n('flux_err')] = res['flux_err']

            if pres['flags'] == 0:
                output[n('T_ratio')] = res['T']/pres['T']

        output['flags'] = flags
        return output

    def _set_fitter(self):
        wpars = self['weight']
        from ngmix.ksigmamom import KSigmaMom

        self._fitter = KSigmaMom(wpars['fwhm'])


class MaxLike(Moments):
    """
    Fit a model via maximum-likelihood.
    """
    def __init__(self, config, rng, nband):
        self.update(config)
        self.rng = rng
        self.nband = nband

        self._setup_fitting()

    def go(self, mbobs_list):
        """
        Fit a model via maximum-likelihood.

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
            else:
                try:
                    res = self.bootstrapper.go(obs=mbobs)
                except BootPSFFailure:
                    res = {
                        'flags': procflags.PSF_FAILURE,
                        'flagstr': procflags.get_name(procflags.PSF_FAILURE),
                    }
                    logger.debug("        fit failed: %s" % res['flagstr'])

            fit_data = self._get_output(obs=mbobs, res=res)

            if res['flags'] == 0:
                self._print_result(fit_data)

            datalist.append(fit_data)

        if len(datalist) == 0:
            return None
        else:
            return eu.numpy_util.combine_arrlist(datalist)

    def _print_result(self, data):
        mess = "        s2n: %g Trat: %g"
        logger.debug(mess % (data['gauss_s2n'][0], data['gauss_T_ratio'][0]))

    def _get_output(self, obs, res):
        npars = 6 + self.nband - 1

        model = 'gauss'
        n = Namer(front=model)

        dt = self._get_dtype(model, npars, flux_nband=self.nband)
        output = np.zeros(1, dtype=dt)

        output['psfrec_flags'] = procflags.NO_ATTEMPT

        output['flags'] = res['flags']
        output[n('flags')] = res['flags']

        if res['flags'] == 0:
            psf_g_avg, psf_T_avg = get_psf_averages(mbobs=obs)

            output['psf_g'] = psf_g_avg
            output['psf_T'] = psf_T_avg

            output[n('s2n')] = res['s2n']
            output[n('pars')] = res['pars']
            output[n('g')] = res['g']
            output[n('g_cov')] = res['g_cov']
            output[n('T')] = res['T']
            output[n('T_err')] = res['T_err']
            output[n('flux')] = res['flux']
            output[n('flux_err')] = res['flux_err']

            output[n('T_ratio')] = res['T']/psf_T_avg

        return output

    def _setup_fitting(self):
        from ngmix.joint_prior import PriorSimpleSep

        self.gal_model = "gauss"
        self.gal_ntry = 2
        self.max_pars = {
            "method": "lm",
            "lm_pars": {
                # "maxfev": 4000,
                "xtol": 5.0e-5,
                "ftol": 5.0e-5,
            }
        }
        sigma_arcsec = 0.1
        cen_prior = ngmix.priors.CenPrior(
            0.0, 0.0,
            sigma_arcsec, sigma_arcsec,
            rng=self.rng,
        )
        g_prior = ngmix.priors.GPriorBA(0.2, rng=self.rng)
        T_prior = ngmix.priors.FlatPrior(-0.1, 1.e+05, rng=self.rng)
        flux_prior = ngmix.priors.FlatPrior(-1000.0, 1.0e+09, rng=self.rng)

        prior = PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            [flux_prior]*self.nband,
        )

        # we use a gaussian for the reconvolved psf
        psf_fitter = ngmix.fitting.Fitter(model='gauss')
        psf_guesser = ngmix.guessers.SimplePSFGuesser(rng=self.rng)

        fitter = ngmix.fitting.Fitter(model=self['model'], prior=prior)

        Tguess = ngmix.moments.fwhm_to_T(0.5)
        guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
            rng=self.rng, T=Tguess, prior=prior,
        )

        psf_runner = ngmix.runners.PSFRunner(
            psf_fitter, guesser=psf_guesser, ntry=4,
        )
        runner = ngmix.runners.Runner(fitter, guesser=guesser, ntry=4)

        self.bootstrapper = ngmix.bootstrap.Bootstrapper(
            runner=runner, psf_runner=psf_runner,
        )


def get_coellip_ngauss(name):
    ngauss = int(name[7:])
    return ngauss


class MaxLikeNgmixv1(Moments):
    """
    Fit a model via maximum-likelihood.
    """
    def __init__(self, config, rng, nband):
        self.update(config)
        self.rng = rng
        self.nband = nband
        self.bootstrapper = Bootstrapper(self.rng, self.nband)

    def go(self, mbobs_list):
        """
        Fit a model via maximum-likelihood.

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
            else:

                try:
                    res = self.bootstrapper.go(mbobs)
                except BootPSFFailure:
                    res = {
                        'flags': procflags.PSF_FAILURE,
                        'flagstr': procflags.get_name(procflags.PSF_FAILURE),
                    }
                    logger.debug("        fit failed: %s" % res['flagstr'])
                except BootGalFailure:
                    res = {
                        'flags': procflags.OBJ_FAILURE,
                        'flagstr': procflags.get_name(procflags.OBJ_FAILURE),
                    }
                    logger.debug("        fit failed: %s" % res['flagstr'])

            fit_data = self._get_output(res)

            if res['flags'] == 0:
                self._print_result(fit_data)

            datalist.append(fit_data)

        if len(datalist) == 0:
            return None
        else:
            return eu.numpy_util.combine_arrlist(datalist)

    def _print_result(self, data):
        mess = "        s2n: %g Trat: %g"
        logger.debug(mess % (data['gauss_s2n'][0], data['gauss_T_ratio'][0]))

    def _get_output(self, res):

        npars = 6 + self.nband - 1

        model = 'gauss'
        n = Namer(front=model)

        dt = self._get_dtype(model, npars, flux_nband=self.nband)
        output = np.zeros(1, dtype=dt)

        output['psfrec_flags'] = procflags.NO_ATTEMPT

        output['flags'] = res['flags']
        output[n('flags')] = res['flags']

        if res['flags'] == 0:
            output['psf_g'] = res['psf_g_avg']
            output['psf_T'] = res['psf_T_avg']

            output[n('s2n')] = res['s2n']
            output[n('pars')] = res['pars']
            output[n('g')] = res['g']
            output[n('g_cov')] = res['g_cov']
            output[n('T')] = res['T']
            output[n('T_err')] = res['T_err']
            output[n('flux')] = res['flux']
            output[n('flux_err')] = res['flux_err']

            output[n('T_ratio')] = res['T']/res['psf_T_avg']

        return output


def fit_all_psfs(mbobs, psf_conf, rng):
    """
    measure all psfs in the input observations and store the results
    in the meta dictionary, and possibly as a gmix for model fits

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to fit
    psf_conf: dict
        Config for  the measurements/fitting
    rng: np.random.RandomState
        The random number generator, used for guessers
    """
    if ngmix.__version__[0:2] == 'v1':
        fit_all_psfs_ngmixv1(mbobs=mbobs, psf_conf=psf_conf, rng=rng)
        return

    if 'coellip' in psf_conf['model']:
        ngauss = get_coellip_ngauss(psf_conf['model'])
        fitter = ngmix.fitting.CoellipFitter(
            ngauss=ngauss, fit_pars=psf_conf['lm_pars'],
        )
        guesser = ngmix.guessers.CoellipPSFGuesser(
            rng=rng, ngauss=ngauss,
        )
    elif psf_conf['model'] == 'wmom':
        fitter = ngmix.gaussmom.GaussMom(fwhm=psf_conf['weight_fwhm'])
        guesser = None
    else:
        fitter = ngmix.fitting.Fitter(
            model=psf_conf['model'], fit_pars=psf_conf['lm_pars'],
        )
        guesser = ngmix.guessers.SimplePSFGuesser(rng=rng)

    runner = ngmix.runners.PSFRunner(
        fitter=fitter, guesser=guesser, ntry=psf_conf.get('ntry', 1),
    )

    for obslist in mbobs:
        assert len(obslist) == 1, 'metadetect is not multi-epoch'

        obs = obslist[0]
        runner.go(obs=obs)

        flags = obs.psf.meta['result']['flags']
        if flags != 0:
            raise BootPSFFailure("failed to measure psfs: %s" % flags)


def fit_all_psfs_ngmixv1(mbobs, psf_conf, rng):
    """
    fit all psfs in the input observations
    """

    for obslist in mbobs:
        for obs in obslist:
            psf_obs = obs.get_psf()
            fit_one_psf_ngmixv1(psf_obs, psf_conf, rng)


def fit_one_psf_ngmixv1(obs, pconf, rng):
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

    obs.meta['result'] = res
    if res['flags'] == 0:
        gmix = psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))

    return res


class Bootstrapper(object):
    def __init__(self, rng, nband):
        self.rng = rng
        self.nband = nband

        self._setup_fitting()

    def go(self, mbobs):

        assert isinstance(mbobs, ngmix.MultiBandObsList)
        self._fit_psfs(mbobs)
        psf_g_avg, psf_T_avg = get_psf_averages(mbobs)

        psf_flux_res = self._fit_gal_psf_flux(mbobs)

        psf_flux = psf_flux_res['psf_flux']

        guesser = self._get_max_guesser(psf_T_avg, psf_flux)
        runner = ngmix.bootstrap.MaxRunner(
            mbobs,
            self.gal_model,
            self.max_pars,
            guesser,
            prior=self.prior,
        )

        runner.go(ntry=self.gal_ntry)

        fitter = runner.fitter

        res = fitter.get_result()

        if res["flags"] != 0:
            raise BootGalFailure("failed to fit galaxy with maxlike")

        res['psf_T_avg'] = psf_T_avg
        res['psf_g_avg'] = psf_g_avg
        return res

    def _fit_psfs(self, mb_obs_list):
        for band, obslist in enumerate(mb_obs_list):
            for i, obs in enumerate(obslist):

                psf_obs = obs.get_psf()
                self._fit_one_psf(psf_obs)

    def _fit_one_psf(self, psf_obs):
        """
        fit the psf and set the gmix if succeeds
        """

        runner = ngmix.bootstrap.PSFRunner(
            psf_obs,
            self.psf_model,
            self.psf_Tguess,
            self.psf_lm_pars,
            rng=self.rng,
        )
        runner.go(ntry=self.psf_ntry)

        psf_fitter = runner.fitter
        res = psf_fitter.get_result()

        if res["flags"] == 0:
            gmix = psf_fitter.get_gmix()
            psf_obs.set_gmix(gmix)
        else:
            raise BootPSFFailure("failed to fit psfs: %s" % str(res))

    def _fit_gal_psf_flux(self, mbo, normalize_psf=True):
        """
        use psf as a template, measure flux (linear)
        """

        nband = len(mbo)

        flags = []
        psf_flux = np.zeros(nband) - 9999.0
        psf_flux_err = np.zeros(nband)

        for i, obs_list in enumerate(mbo):

            if len(obs_list) == 0:
                raise BootPSFFailure("no epochs for band %d" % i)

            if not obs_list[0].has_psf_gmix():
                raise RuntimeError("you need to fit the psfs first")

            fitter = ngmix.fitting.TemplateFluxFitter(
                obs_list, do_psf=True, normalize_psf=normalize_psf,
            )
            fitter.go()

            res = fitter.get_result()
            tflags = res["flags"]
            flags.append(tflags)

            if tflags == 0:

                psf_flux[i] = res["flux"]
                psf_flux_err[i] = res["flux_err"]

            else:
                print("failed to fit psf flux for band", i)

        return {
            "flags": flags,
            "psf_flux": psf_flux,
            "psf_flux_err": psf_flux_err,
        }

    def _get_max_guesser(self, psf_T, psf_flux):
        """
        get a guesser that uses the psf T and galaxy psf flux to
        generate a guess, drawing from priors on the other parameters
        """

        guesser = ngmix.guessers.TFluxAndPriorGuesser(
            psf_T, psf_flux, self.prior, scaling="linear",
        )
        return guesser

    def _setup_fitting(self):
        from ngmix.joint_prior import PriorSimpleSep

        self.gal_model = "gauss"
        self.gal_ntry = 2
        self.max_pars = {
            "method": "lm",
            "lm_pars": {
                # "maxfev": 4000,
                "xtol": 5.0e-5,
                "ftol": 5.0e-5,
            }
        }
        sigma_arcsec = 0.1
        cen_prior = ngmix.priors.CenPrior(
            0.0, 0.0,
            sigma_arcsec, sigma_arcsec,
            rng=self.rng,
        )
        g_prior = ngmix.priors.GPriorBA(0.2, rng=self.rng)
        T_prior = ngmix.priors.FlatPrior(-0.1, 1.e+05, rng=self.rng)
        flux_prior = ngmix.priors.FlatPrior(-1000.0, 1.0e+09, rng=self.rng)

        self.prior = PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            [flux_prior]*self.nband,
        )

        self.psf_model = "gauss"
        self.psf_Tguess = ngmix.moments.fwhm_to_T(0.9)
        self.psf_lm_pars = {
            "maxfev": 4000,
            "xtol": 5.0e-5,
            "ftol": 5.0e-5,
        }
        self.psf_ntry = 4


def get_psf_averages(mbobs):
    """
    get weighted averages of g and T using the max value from the weight maps
    as the weight
    """
    Tsum = 0.0
    g1sum = 0.0
    g2sum = 0.0
    wsum = 0.0

    for obslist in mbobs:
        for obs in obslist:
            wt = np.max(obs.weight)

            g1, g2, T = obs.psf.gmix.get_g1g2T()

            wsum += wt
            Tsum += wt*T
            g1sum += wt*g1
            g2sum += wt*g2

    if wsum <= 0.0:
        g1 = g2 = T = 9999.0, 9999.0, 9999.0
    else:
        T = Tsum/wsum
        g1 = g1sum/wsum
        g2 = g2sum/wsum

    return np.array([g1, g2]), T
