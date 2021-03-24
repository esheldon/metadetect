import logging
import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure
import esutil as eu
from . import detect
from . import fitting
from . import procflags
from . import shearpos
from .mfrac import measure_mfrac

logger = logging.getLogger(__name__)


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
            max (if running a max like fitter)
            fofs (if running MOF)
            mof (if running MOF)

    mbobs: ngmix.MultiBandObsList
        We will do detection and measurements on these images
    rng: numpy.random.RandomState
        Random number generator
    """
    def __init__(self, config, mbobs, rng, show=False):

        self._show = show

        self._set_config(config)
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.rng = rng

        if 'psf' in config:
            self._fit_original_psfs()

        self._set_fitter()

        self._set_ormask()
        self._set_mfrac()

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

        wstar = np.where((ormask & self['star_flags']) != 0)
        wtapebump = np.where((ormask & self['tapebump_flags']) != 0)
        wspline_interp = np.where((ormask & self['spline_interp_flags']) != 0)
        wnoise_interp = np.where((ormask & self['noise_interp_flags']) != 0)
        wimperfect = np.where((ormask & self['imperfect_flags']) != 0)

        self.star_frac = wstar[0].size/ormask.size
        self.tapebump_frac = wtapebump[0].size/ormask.size
        self.noise_interp_frac = wnoise_interp[0].size/ormask.size
        self.spline_interp_frac = wspline_interp[0].size/ormask.size
        self.imperfect_frac = wimperfect[0].size/ormask.size

    def _set_mfrac(self):
        """
        set the masked fraction image, averaged over all bands
        """
        wgts = []
        mfrac = np.zeros_like(self.mbobs[0][0].image)
        for band, obslist in enumerate(self.mbobs):
            nepoch = len(obslist)
            assert nepoch == 1, 'expected 1 epoch, got %d' % nepoch

            obs = obslist[0]
            wgt = np.median(obs.weight)
            if hasattr(obs, "mfrac"):
                mfrac += (obs.mfrac * wgt)
            wgts.append(wgt)

        self.mfrac = mfrac / np.sum(wgts)

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
        self['model'] = self.get('model', 'wmom')

        if self['model'] == 'wmom':
            self._fitter = fitting.Moments(self, self.rng)
        elif self['model'] == 'gauss':
            if ngmix.__version__[1] == '1':
                self._fitter = fitting.MaxLikeNgmixv1(
                    self, self.rng, self.nband,
                )
            else:
                self._fitter = fitting.MaxLike(self, self.rng, self.nband)
        else:
            raise ValueError("bad model: '%s'" % self['model'])

    def _measure(self, mbobs, shear_str):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements
        """

        medsifier = self._do_detect(mbobs)
        if self._show:
            import descwl_coadd.vis
            descwl_coadd.vis.show_image(medsifier.seg)

        mbm = medsifier.get_multiband_meds()

        mbobs_list = mbm.get_mbobs_list()

        res = self._fitter.go(mbobs_list)

        if res is not None:
            res = self._add_positions_and_psf(medsifier.cat, res, shear_str)

        return res

    def _add_positions_and_psf(self, cat, res, shear_str):
        """
        add catalog positions to the result
        """

        new_dt = [
            ('sx_row', 'f4'),
            ('sx_col', 'f4'),
            ('sx_row_noshear', 'f4'),
            ('sx_col_noshear', 'f4'),
            ('ormask', 'i4'),
            ('mfrac', 'f4'),
            ('star_frac', 'f4'),  # these are for the whole image, redundant
            ('tapebump_frac', 'f4'),
            ('spline_interp_frac', 'f4'),
            ('noise_interp_frac', 'f4'),
            ('imperfect_frac', 'f4'),
        ]
        newres = eu.numpy_util.add_fields(
            res,
            new_dt,
        )

        if hasattr(self, 'psf_stats'):
            newres['psfrec_flags'][:] = self.psf_stats['flags']
            newres['psfrec_g'][:, 0] = self.psf_stats['g1']
            newres['psfrec_g'][:, 1] = self.psf_stats['g2']
            newres['psfrec_T'][:] = self.psf_stats['T']

        newres['star_frac'][:] = self.star_frac
        newres['tapebump_frac'][:] = self.tapebump_frac
        newres['spline_interp_frac'][:] = self.spline_interp_frac
        newres['noise_interp_frac'][:] = self.noise_interp_frac
        newres['imperfect_frac'][:] = self.imperfect_frac

        if cat.size > 0:
            obs = self.mbobs[0][0]

            newres['sx_col'] = cat['x']
            newres['sx_row'] = cat['y']

            rows_noshear, cols_noshear = shearpos.unshear_positions(
                newres['sx_row'],
                newres['sx_col'],
                shear_str,
                obs,  # an example for jacobian and image shape
                # default is 0.01 but make sure to use the passed in default
                # if needed
                step=self['metacal'].get("step", shearpos.DEFAULT_STEP),
            )

            newres['sx_row_noshear'] = rows_noshear
            newres['sx_col_noshear'] = cols_noshear

            dims = obs.image.shape
            rclip = _clip_and_round(rows_noshear, dims[0])
            cclip = _clip_and_round(cols_noshear, dims[1])

            if 'ormask_region' in self and self['ormask_region'] > 1:
                logger.debug('ormask_region: %s' % self['ormask_region'])
                for ind in range(cat.size):
                    lr = int(min(
                        dims[0]-1,
                        max(0, rclip[ind] - self['ormask_region'])))
                    ur = int(min(
                        dims[0]-1,
                        max(0, rclip[ind] + self['ormask_region'])))

                    lc = int(min(
                        dims[1]-1,
                        max(0, cclip[ind] - self['ormask_region'])))
                    uc = int(min(
                        dims[1]-1,
                        max(0, cclip[ind] + self['ormask_region'])))

                    newres['ormask'][ind] = np.bitwise_or.reduce(
                        self.ormask[lr:ur+1, lc:uc+1], axis=None)
            else:
                newres['ormask'] = self.ormask[rclip, cclip]

            if np.any(self.mfrac > 0):
                newres["mfrac"] = measure_mfrac(
                    mfrac=self.mfrac,
                    cat=cat,
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                    step=self['metacal'].get("step", shearpos.DEFAULT_STEP),
                )
            else:
                newres["mfrac"] = 0

        return newres

    def _do_detect(self, mbobs):
        """
        use a MEDSifier to run detection
        """
        return detect.MEDSifier(
            mbobs=mbobs,
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

        if self._show:
            import descwl_coadd.vis

            orig_mbobs = self.mbobs

            for mtype, mbobs in odict.items():
                for band in range(len(mbobs)):

                    obslist = mbobs[band]
                    orig_obslist = orig_mbobs[band]

                    for iobs in range(len(obslist)):
                        obs = obslist[iobs]
                        orig_obs = orig_obslist[iobs]

                        descwl_coadd.vis.show_images(
                            [
                                obs.image,
                                obs.bmask,
                                obs.weight,
                                orig_obs.noise,
                            ],
                            title=mtype,
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
                    res = obs.psf.meta['result']
                    T = res['T']
                    if 'e' in res:
                        g1, g2 = res['e']
                    else:
                        g1, g2 = res['g']

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

    np.rint(vals, out=vals)
    vals.clip(min=0, max=dim-1, out=vals)

    return vals.astype('i4')
