"""

TODO
    - add logging
    - tests
    - more TODO are in the code
"""
import copy
import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure
# import lsst.log
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel
import lsst.afw.image as afw_image
from .lsst_measure import detect_and_measure, subtract_sky_mbobs
from . import shearpos
from .mfrac import measure_mfrac
from . import procflags
from . import fitting
from .defaults import DEFAULT_LOGLEVEL, DEFAULT_MDET_CONFIG


def run_metadetect(config, mbobs, rng, show=False, loglevel=DEFAULT_LOGLEVEL):
    """
    config: dict
        Configuration for the fitter, metacal, psf, detect, deblend,
        subtract_sky, etc.
    mbobs: ngmix.MultiBandObsList
        The observations to process
    rng: np.random.RandomState
        Random number generator
    show: bool
        if True images will be shown
    loglevel: str
        Default 'INFO'

    Returns
    -------
    result dict
        This is keyed by shear string 'noshear', '1p', ... or None if there was
        a problem doing the metacal steps; this only happens if the setting
        metacal_psf is set to 'fitgauss' and the fitting fails
    """
    config = get_config(config)
    metacal_config = {
        'use_noise_image': True,
        'psf': config['metacal_psf']
    }
    stamp_size = get_stamp_size(meas_type=config['meas_type'])

    if config['subtract_sky']:
        subtract_sky_mbobs(mbobs=mbobs, thresh=config['detect_thresh'])

    # TODO we get psf stats for the entire coadd, not location dependent
    # for each object on original image
    psf_stats = fit_original_psfs(config=config, mbobs=mbobs, rng=rng)

    fitter = get_fitter(config)
    ormask, bmask = get_ormask_and_bmask(mbobs)
    mfrac = get_mfrac(mbobs)

    odict = get_all_metacal(
        metacal_config=metacal_config, mbobs=mbobs, rng=rng, show=show,
    )

    if odict is None:
        result = None
    else:
        result = {}
        for shear_str, mbobs in odict.items():
            assert len(mbobs) == 1, 'no multiband for now'
            assert len(mbobs[0]) == 1, 'no multiepoch'
            obs = mbobs[0][0]
            exposure = obs.exposure
            res = detect_and_measure(
                exposure=exposure,
                fitter=fitter,
                stamp_size=stamp_size,
                thresh=config['detect_thresh'],
                use_deblended_stamps=config['use_deblended_stamps'],
                loglevel=loglevel,
            )

            if res is not None:
                obs = mbobs[0][0]
                add_noshear_pos(config, res, shear_str, obs)
                add_mfrac(config, mfrac, res, obs)
                add_ormask(ormask, res)
                add_original_psf(psf_stats, res)

            result[shear_str] = res

    return result


def add_noshear_pos(config, res, shear_str, obs):
    rows_noshear, cols_noshear = shearpos.unshear_positions(
        res['row'],
        res['col'],
        shear_str,
        obs,  # an example for jacobian and image shape
    )
    res['row_noshear'] = rows_noshear
    res['col_noshear'] = cols_noshear


def add_mfrac(config, mfrac, res, obs):
    if np.any(mfrac > 0):
        # we are using the positions with the metacal shear removed for
        # this.
        res['mfrac'] = measure_mfrac(
            x=res['col_noshear'],
            y=res['row_noshear'],
            box_sizes=res['box_size'],
            obs=obs,
            fwhm=config.get('mfrac_fwhm', None),
        )
    else:
        res['mfrac'] = 0


def add_ormask(ormask, res):
    for i in range(res.size):
        res['ormask'][i] = ormask[
            int(res['row'][i]), int(res['col'][i]),
        ]


def add_original_psf(psf_stats, res):

    res['psfrec_flags'][:] = psf_stats['flags']
    res['psfrec_g'][:, 0] = psf_stats['g1']
    res['psfrec_g'][:, 1] = psf_stats['g2']
    res['psfrec_T'][:] = psf_stats['T']


def get_fitter(config):
    if 'model' not in config:
        if 'fitter' not in config:
            fitter_type = 'wmom'
        else:
            fitter_type = config['fitter']
    else:
        fitter_type = config['model']

    if fitter_type == 'wmom':
        fitter = ngmix.gaussmom.GaussMom(fwhm=config['weight_fwhm'])
    elif fitter_type == 'ksigma':
        fitter = ngmix.ksigmamom.KSigmaMom(fwhm=config['weight_fwhm'])
    else:
        raise ValueError("bad fitter type: '%s'" % fitter_type)

    return fitter


def get_all_metacal(metacal_config, mbobs, rng, show=False):
    """
    get the sheared versions of the observations

    call the parent and then add in the stack exposure with image copied
    in, modify the variance and set the new psf
    """

    orig_mbobs = mbobs

    try:
        odict = ngmix.metacal.get_all_metacal(
            orig_mbobs,
            rng=rng,
            **metacal_config,
        )
    except BootPSFFailure:
        # this can happen if we were using psf='fitgauss'
        return None

    # make sure .exposure is set for each obs
    for mtype, mbobs in odict.items():
        for band in range(len(mbobs)):

            obslist = mbobs[band]
            orig_obslist = orig_mbobs[band]

            for iobs in range(len(obslist)):
                obs = obslist[iobs]
                orig_obs = orig_obslist[iobs]

                exp = copy.deepcopy(orig_obs.coadd_exp)
                exp.image.array[:, :] = obs.image

                # we ran fixnoise, need to update variance plane
                exp.variance.array[:, :] = exp.variance.array[:, :]*2

                psf_image = obs.psf.image
                stack_psf = KernelPsf(
                    FixedKernel(
                        afw_image.ImageD(psf_image.astype(float))
                    )
                )
                exp.setPsf(stack_psf)
                obs.exposure = exp

                if show:
                    import descwl_coadd.vis
                    descwl_coadd.vis.show_image_and_mask(exp)
                    input('hit a key')

    return odict


def get_ormask_and_bmask(mbobs):
    """
    set the ormask and bmask, ored from all epochs
    """

    for band, obslist in enumerate(mbobs):
        nepoch = len(obslist)
        assert nepoch == 1, 'expected 1 epoch, got %d' % nepoch

        obs = obslist[0]

        if band == 0:
            ormask = obs.ormask.copy()
            bmask = obs.bmask.copy()
        else:
            ormask |= obs.ormask
            bmask |= obs.bmask

    return ormask, bmask


def get_mfrac(mbobs):
    """
    set the masked fraction image, averaged over all bands
    """
    wgts = []
    mfrac = np.zeros_like(mbobs[0][0].image)
    for band, obslist in enumerate(mbobs):
        nepoch = len(obslist)
        assert nepoch == 1, 'expected 1 epoch, got %d' % nepoch

        obs = obslist[0]
        wgt = np.median(obs.weight)
        if hasattr(obs, "mfrac"):
            mfrac += (obs.mfrac * wgt)
        wgts.append(wgt)

    mfrac = mfrac / np.sum(wgts)
    return mfrac


def fit_original_psfs(config, mbobs, rng):
    """
    fit the original psfs and get the mean g1,g2,T across
    all bands

    This can fail and flags will be set, but we proceed
    """

    psf_config = {'model': 'admom', 'ntry': 4}
    try:
        fitting.fit_all_psfs(mbobs, psf_config, rng)

        g1sum = 0.0
        g2sum = 0.0
        Tsum = 0.0
        wsum = 0.0

        for obslist in mbobs:
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

    return {
        'flags': flags,
        'g1': g1,
        'g2': g2,
        'T': T,
    }


def get_config(config):

    for key in config:
        if key not in DEFAULT_MDET_CONFIG:
            raise ValueError('bad key in mdet config: %s' % key)

    config_new = copy.deepcopy(DEFAULT_MDET_CONFIG)
    config_new.update(copy.deepcopy(config))

    return config_new


def get_stamp_size(meas_type):
    if meas_type == 'wmom':
        stamp_size = 32
    elif meas_type == 'ksigma':
        # TODO figure this out
        stamp_size = 64
    else:
        raise ValueError('bad meas type: %s' % meas_type)

    return stamp_size


'''
class LSSTMetadetect(BaseLSSTMetadetect):
    """
    Metadetect for LSST

    Parameters
    ----------
    config: dict
        The configuration
    mbobs: ngmix.MultiBandObsList
        The observations
    rng: np.random.RandomState
        the random state
    show: bool, optional
        If True will show images
    loglevel: str, optional
        Defaults to 'INFO'
    """

    name = 'LSSTMetadetect'

    def __init__(
        self, config, mbobs, rng,
        show=False, loglevel=DEFAULT_LOGLEVEL,
    ):

        self._set_logger(loglevel)

        super().__init__(config=config, mbobs=mbobs, rng=rng, show=show)

        subtract_sky = self.get('subtract_sky', False)
        if subtract_sky:
            subtract_sky_mbobs(mbobs=self.mbobs, thresh=self['detect']['thresh'])

    def _set_logger(self, loglevel):
        self.loglevel = loglevel
        self.log = lsst.log.Log.getLogger(self.name)
        self.log.setLevel(getattr(lsst.log, self.loglevel.upper()))

    def _get_all_metacal(self):
        """
        get the sheared versions of the observations

        call the parent and then add in the stack exposure with image copied
        in, modify the variance and set the new psf
        """

        did_fixnoise = self['metacal'].get('fixnoise', True)
        orig_mbobs = self.mbobs
        odict = super()._get_all_metacal()
        for mtype, mbobs in odict.items():
            for band in range(len(mbobs)):

                obslist = mbobs[band]
                orig_obslist = orig_mbobs[band]

                for iobs in range(len(obslist)):
                    obs = obslist[iobs]
                    orig_obs = orig_obslist[iobs]

                    exp = copy.deepcopy(orig_obs.coadd_exp)
                    exp.image.array[:, :] = obs.image
                    if did_fixnoise:
                        exp.variance.array[:, :] = exp.variance.array[:, :]*2

                    psf_image = obs.psf.image
                    stack_psf = KernelPsf(
                        FixedKernel(
                            afw_image.ImageD(psf_image.astype(float))
                        )
                    )
                    exp.setPsf(stack_psf)
                    obs.exposure = exp

                    if self._show:
                        import descwl_coadd.vis
                        descwl_coadd.vis.show_image_and_mask(exp)
                        input('hit a key')

        return odict

    def _do_detect(self, mbobs):
        return LSSTMEDSifier(
            mbobs=mbobs,
            meds_config=self['meds'],
            thresh=self['detect']['thresh'],
            loglevel=self.loglevel,
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
        if res is not None:
            res = self._add_positions_and_psf(medsifier, res, shear_str, mbobs_list)

        return res

    def _add_positions_and_psf(self, medsifier, res, shear_str, mbobs_list):
        """
        TODO add positionis etc.
        """

        new_dt = [
            ('box_size', 'i4'),
            ('row0', 'i4'),  # bbox row start
            ('col0', 'i4'),  # bbox col start
            ('row', 'f4'),  # row in image. Use row0 to get to global pixel coords
            ('col', 'f4'),  # col in image. Use col0 to get to global pixel coords
            ('row_noshear', 'f4'),  # noshear row
            ('col_noshear', 'f4'),  # noshear col
            ('ormask', 'i4'),
            ('mfrac', 'f4'),
        ]
        newres = eu.numpy_util.add_fields(
            res,
            new_dt,
        )

        newres['psfrec_flags'][:] = self.psf_stats['flags']
        newres['psfrec_g'][:, 0] = self.psf_stats['g1']
        newres['psfrec_g'][:, 1] = self.psf_stats['g2']
        newres['psfrec_T'][:] = self.psf_stats['T']

        sources = medsifier.sources
        det_exp = medsifier.det_exp
        bbox = det_exp.getBBox()

        for i, rec in enumerate(sources):
            orig_cen = det_exp.getWcs().skyToPixel(rec.getCoord())
            if np.isnan(orig_cen.getY()):
                self.log.debug('falling back on integer location')
                # fall back to integer pixel location
                peak = rec.getFootprint().getPeaks()[0]
                orig_cen = peak.getI()

            # these might be crazy if bbox was odd, but we will
            # set a flag and bbox will be negative
            newres['row0'][i] = bbox.getBeginY()
            newres['col0'][i] = bbox.getBeginX()
            newres['row'][i] = orig_cen.getY() - newres['row0'][i]
            newres['col'][i] = orig_cen.getX() - newres['col0'][i]

            if len(mbobs_list[i]) > 0 and len(mbobs_list[i][0]) > 0:
                newres['box_size'][i] = mbobs_list[i][0][0].image.shape[0]

                newres['ormask'][i] = self.ormask[
                    int(newres['row'][i]), int(newres['col'][i]),
                ]
            else:
                # the above happens when the bbox is crazy
                newres['flags'][i] |= procflags.BAD_BBOX
                newres['box_size'][i] = -9999

        if len(sources) > 0:

            obs = self.mbobs[0][0]

            rows_noshear, cols_noshear = shearpos.unshear_positions(
                newres['row'],
                newres['col'],
                shear_str,
                obs,  # an example for jacobian and image shape
                # default is 0.01 but make sure to use the passed in default
                # if needed
                step=self['metacal'].get("step", shearpos.DEFAULT_STEP),
            )
            newres['row_noshear'] = rows_noshear
            newres['col_noshear'] = cols_noshear

            if np.any(self.mfrac > 0):
                # we are using the positions with the metacal shear removed for
                # this.
                newres["mfrac"] = measure_mfrac(
                    mfrac=self.mfrac,
                    x=newres["col_noshear"],
                    y=newres["row_noshear"],
                    box_sizes=newres["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                )
            else:
                newres["mfrac"] = 0

                # mark bad bboxes as masked as they are when the
                # measure_mfrac code is run
                w, = np.where(newres['flags'] & procflags.BAD_BBOX != 0)
                newres['mfrac'][w] = 1.0

        return newres

    def _set_config(self, config):
        """
        set the config, dealing with defaults
        """

        self.update(config)
        assert 'metacal' in self, \
            'metacal setting must be present in config'
        # assert 'meds' in self, \
        #     'meds setting must be present in config'


#
# new code
#



class LSSTDeblendMetadetect(LSSTMetadetect):
    """
    Metadetect for LSST with deblending at the measurement
    level

    Parameters
    ----------
    config: dict
        The configuration
    mbobs: ngmix.MultiBandObsList
        The observations
    rng: np.random.RandomState
        the random state
    show: bool, optional
        If True will show images
    loglevel: str, optional
        Defaults to 'INFO'
    """

    name = 'LSSTDeblendMetadetect'

    def __init__(
        self, config, mbobs, rng,
        show=False, loglevel='INFO',
    ):
        super().__init__(
            config=config, mbobs=mbobs, rng=rng, show=show,
        )

        self._set_weight()

    def _set_fitter(self):
        pass

    def _do_detect(self, mbobs):
        raise NotImplementedError('no do detect for this class')

    def _measure(self, mbobs, shear_str):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements
        """

        res = measure_weighted_moments(
            mbobs=mbobs,
            weight=self.weight,
            thresh=self['detect']['thresh'],
        )

        if res is not None:
            obs = mbobs[0][0]
            self._add_noshear_pos(res, shear_str, obs)
            self._add_mfrac(res, obs)
            self._add_ormask(res)
            res = self._add_original_psf(res, shear_str)

        return res

    def _add_ormask(self, res):
        for i in range(res.size):
            res['ormask'][i] = self.ormask[
                int(res['row'][i]), int(res['col'][i]),
            ]

    def _add_noshear_pos(self, res, shear_str, obs):
        rows_noshear, cols_noshear = shearpos.unshear_positions(
            res['row'],
            res['col'],
            shear_str,
            obs,  # an example for jacobian and image shape
            # default is 0.01 but make sure to use the passed in default
            # if needed
            step=self['metacal'].get("step", shearpos.DEFAULT_STEP),
        )
        res['row_noshear'] = rows_noshear
        res['col_noshear'] = cols_noshear

    def _add_mfrac(self, res, obs):
        if np.any(self.mfrac > 0):
            # we are using the positions with the metacal shear removed for
            # this.
            res["mfrac"] = measure_mfrac(
                mfrac=self.mfrac,
                x=res["col_noshear"],
                y=res["row_noshear"],
                box_sizes=res["box_size"],
                obs=obs,
                fwhm=self.get("mfrac_fwhm", None),
            )
        else:
            res["mfrac"] = 0

    def _add_original_psf(self, res, shear_str):
        """
        TODO add ormask etc.
        """

        res['psfrec_flags'][:] = self.psf_stats['flags']
        res['psfrec_g'][:, 0] = self.psf_stats['g1']
        res['psfrec_g'][:, 1] = self.psf_stats['g2']
        res['psfrec_T'][:] = self.psf_stats['T']

        return res

    def _set_weight(self):
        wpars = self['weight']

        T = ngmix.moments.fwhm_to_T(wpars['fwhm'])  # noqa

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
'''
