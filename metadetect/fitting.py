import ngmix

def fit_all_psfs(mbobs, psf_conf):
    """
    fit all psfs in the input observations
    """
    fitter=AllPSFFitter(mbobs, psf_conf)
    fitter.go()

class AllPSFFitter(object):
    """
    fit all psfs using the specified config
    """
    def __init__(self, mbobs, psf_conf):
        self.mbobs=mbobs
        self.psf_conf=psf_conf

    def go(self):
        for obslist in self.mbobs:
            for obs in obslist:
                psf_obs = obs.get_psf()
                fit_one_psf(psf_obs, self.psf_conf)

def fit_one_psf(obs, pconf, rng):
    Tguess=4.0*obs.jacobian.get_scale()**2

    if 'coellip' in pconf['model']:
        ngauss=ngmix.bootstrap.get_coellip_ngauss(pconf['model'])
        runner=ngmix.bootstrap.PSFRunnerCoellip(
            obs,
            Tguess,
            ngauss,
            pconf['lm_pars'],
            rng=rng,
        )

    else:
        runner=ngmix.bootstrap.PSFRunner(
            obs,
            pconf['model'],
            Tguess,
            pconf['lm_pars'],
            rng=rng,
        )

    runner.go(ntry=pconf['ntry'])

    psf_fitter = runner.fitter
    res=psf_fitter.get_result()
    obs.update_meta_data({'fitter':psf_fitter})

    if res['flags']==0:
        gmix=psf_fitter.get_gmix()
        obs.set_gmix(gmix)
    else:
        raise BootPSFFailure("failed to fit psfs: %s" % str(res))


