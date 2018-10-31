import ngmix

def do_metadetect(config, mbobs, rng):
    """
    meta-detect on the multi-band observations.
    """
    md=Metadetect(config, mbobs, rng)
    md.go()
    return md.result

class Metadetect(dict):
    """
    meta-detect on the multi-band observations.

    parameters
    ----------
    config: dict
        Configuration dictionary. Possible entries are
            metacal
            weight (if calculating weighted moments)
            max: (if running a max like fitter)
            fofs (if running MOF)
            mof (if running MOF)

    """
    def __init__(self, config, mbobs, rng):
        self.update(config)
        self.mbobs=mbobs
        self.rng=rng

    @property
    def result(self):
        """
        get the result dict, keyed by the metacal type such
        as 'noshear', '1p', '1m', '2p', '2m'
        """
        if not hasattr(self,'_result'):
            raise RuntimeError('run go() first')

        return self._result

    def go(self):
        """
        make sheared versions of the images, run detection and measurements on each
        """
        odict=self._get_all_metacal()

        self._result = {}
        for key, mbobs in odict.items():
            self._result[key] = self._measure(mbobs)

    def _measure(self, mbobs):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements
        """
        return {}

    def _get_all_metacal(self):
        """
        get the sheared versions of the observations
        """
        odict = ngmix.metacal.get_all_metacal(
            self.mbobs,
            rng=sim.rng,
            **metacal_pars
        )

        return odict


