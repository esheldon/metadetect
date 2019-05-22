import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure
import esutil as eu
from . import detect
from . import fitting
from . import procflags
from . import shearpos


def do_metadetect(config, mbobs, rng):
    """
    meta-detect on the multi-band observations.  For parameters, see
    docs on the Metadetect class
    """
    md = Metadetect(config, mbobs, rng)
    md.go()
    return md.result


class Metadetect(dict):
    """
    meta-detect on the multi-band observations.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Possible entries are
            metacal
            weight (if calculating weighted moments)
            max: (if running a max like fitter)
            fofs (if running MOF)
            mof (if running MOF)

    mbobs: ngmix.MultiBandObsList
        We will do detection and measurements on these images
    rng: numpy.random.RandomState
        Random number generator
    """
    def __init__(self, config, mbobs, rng):
        self._set_config(config)
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.rng = rng

        self._fit_original_psfs()

        self._set_fitter()

        self._set_ormask()

    def _set_ormask(self):
        """
        set the ormask, ored from all ormasks
        """

        for band, obslist in enumerate(self.mbobs):
            nepoch = len(obslist)
            assert nepoch == 1, 'expected 1 epoch, got %d' % nepoch

            obs = obslist[0]

            if band == 0:
                ormask = obs.ormask.copy()
            else:
                ormask |= obs.ormask

        self.ormask = ormask

    @property
    def result(self):
        """
        get the result dict, keyed by the metacal type such
        as 'noshear', '1p', '1m', '2p', '2m'
        """
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        return self._result

    def go(self):
        """
        make sheared versions of the images, run detection and measurements on
        each
        """
        try:
            odict = self._get_all_metacal()
        except BootPSFFailure:
            odict = None

        if odict is None:
            self._result = None
        else:
            self._result = {}
            for shear_str, mbobs in odict.items():
                self._result[shear_str] = self._measure(mbobs, shear_str)

    def _set_fitter(self):
        """
        set the fitter to be used
        """
        self._fitter = fitting.Moments(
            self,
            self.rng,
        )

    def _measure(self, mbobs, shear_str):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements
        """

        medsifier = self._do_detect(mbobs)
        mbm = medsifier.get_multiband_meds()

        mbobs_list = mbm.get_mbobs_list()

        res = self._fitter.go(mbobs_list)

        res = self._add_positions_and_psf(medsifier.cat, res, shear_str)
        return res

    def _add_positions_and_psf(self, cat, res, shear_str):
        """
        add catalog positions to the result
        """

        res['psfrec_flags'][:] = self.psf_stats['flags']
        res['psfrec_g'][:, 0] = self.psf_stats['g1']
        res['psfrec_g'][:, 1] = self.psf_stats['g2']
        res['psfrec_T'][:] = self.psf_stats['T']

        if cat.size > 0:
            obs = self.mbobs[0][0]

            new_dt = [
                ('sx_row', 'f4'),
                ('sx_col', 'f4'),
                ('sx_row_noshear', 'f4'),
                ('sx_col_noshear', 'f4'),
                ('ormask', 'i4'),
            ]
            newres = eu.numpy_util.add_fields(
                res,
                new_dt,
            )

            newres['sx_col'] = cat['x']
            newres['sx_row'] = cat['y']

            rows_noshear, cols_noshear = shearpos.unshear_positions(
                newres['sx_row'],
                newres['sx_col'],
                shear_str,
                obs,  # an example for jacobian and image shape
            )

            newres['sx_row_noshear'] = rows_noshear
            newres['sx_col_noshear'] = cols_noshear

            dims = obs.image.shape
            rclip = _clip_and_round(rows_noshear, dims[0])
            cclip = _clip_and_round(cols_noshear, dims[1])

            newres['ormask'] = self.ormask[rclip, cclip]

        else:
            newres = res

        return newres

    def _do_detect(self, mbobs):
        """
        use a MEDSifier to run detection
        """
        return detect.MEDSifier(
            mbobs,
            sx_config=self['sx'],
            meds_config=self['meds'],
        )

    def _get_all_metacal(self):
        """
        get the sheared versions of the observations
        """

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

    def _fit_original_psfs(self):
        """
        fit the original psfs and get the mean g1,g2,T across
        all bands

        This can fail and flags will be set, but we proceed
        """

        try:
            fitting.fit_all_psfs(self.mbobs, self['psf'], self.rng)

            g1sum = 0.0
            g2sum = 0.0
            Tsum = 0.0
            wsum = 0.0

            for obslist in self.mbobs:
                for obs in obslist:
                    wt = obs.weight.max()
                    g1, g2, T = obs.psf.gmix.get_g1g2T()

                    g1sum += g1*wt
                    g2sum += g2*wt
                    Tsum += T*wt
                    wsum += wt

            if wsum <= 0.0:
                raise BootPSFFailure('zero weights, could not '
                                     'get mean psf properties')
            g1 = g1sum/wsum
            g2 = g2sum/wsum
            T = Tsum/wsum

            flags = 0

        except BootPSFFailure:
            flags = procflags.PSF_FAILURE
            g1 = -9999.0
            g2 = -9999.0
            T = -9999.0

        self.psf_stats = {
            'flags': flags,
            'g1': g1,
            'g2': g2,
            'T': T,
        }


def _clip_and_round(vals_in, dim):
    """
    clip values and round to nearest integer
    """

    vals = vals_in.copy()

    vals.clip(min=0, max=dim-1, out=vals)
    np.rint(vals, out=vals)

    return vals.astype('i4')
