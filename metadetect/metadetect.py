import ngmix
from . import detect
from . import fitting
from . import defaults

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
        self._set_config(config)
        self.mbobs=mbobs
        self.nband = len(mbobs)
        self.rng=rng

        self._set_fitter()

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


    def _set_fitter(self):
        """
        set the fitter to be used
        """
        self._fitter=fitting.Moments(
            self,
            self.rng,
        )

    def _measure(self, mbobs):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements
        """

        mer = self._do_detect(mbobs)
        mbm=mer.get_multiband_meds()

        mbobs_list = mbm.get_mbobs_list()

        res=self._fitter.go(mbobs_list)
        return {}

    def _do_detect(self, mbobs):
        """
        use a MEDSifier to run detection
        """
        mer=detect.MEDSifier(
            mbobs,
            sx_config=self['sx'],
            meds_config=self['meds'],
        )
        return mer



    def _get_all_metacal(self):
        """
        get the sheared versions of the observations
        """

        if self['metacal'].get('symmetrize_psf',False):
            assert 'psf' in self,'need psf fitting for symmetrize_psf'
            fitting.fit_all_psfs(self.mbobs, self['psf'], self.rng)

        odict = ngmix.metacal.get_all_metacal(
            self.mbobs,
            rng=self.rng,
            **self['metacal']
        )

        return odict

    def _set_config(self, config):
        """
        set the config, dealing with defaults
        """

        self.update(config)
        assert 'metacal' in self, \
            'metacal setting must be present in config'
        assert 'sx' in self, \
            'sx setting must be present in config'
        assert 'meds' in self, \
            'meds setting must be present in config'
