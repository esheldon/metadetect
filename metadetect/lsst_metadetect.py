import copy
import numpy as np
from .base import BaseLSSTMetadetect
import ngmix
import esutil as eu
import lsst.log
from lsst.meas.algorithms import KernelPsf
from lsst.afw.math import FixedKernel
import lsst.afw.image as afw_image
from .lsst_medsifier import LSSTMEDSifier
from .lsst_measure import measure_weighted_moments
from . import shearpos
from .mfrac import measure_mfrac
from . import procflags


class LSSTMetadetect(BaseLSSTMetadetect):
    def __init__(self, *args, **kw):
        loglevel = kw.pop('loglevel', 'info').upper()

        super().__init__(*args, **kw)

        self.log = lsst.log.Log.getLogger("LSSTMetadetect")
        self.log.setLevel(getattr(lsst.log, loglevel))
        self.loglevel = loglevel

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

                # mark bad bboxes as masked
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


class LSSTDeblendMetadetect(LSSTMetadetect):
    def __init__(self, *args, **kw):
        loglevel = kw.pop('loglevel', 'info').upper()

        super(LSSTMetadetect, self).__init__(*args, **kw)

        self.log = lsst.log.Log.getLogger("LSSTDeblendMetadetect")
        self.log.setLevel(getattr(lsst.log, loglevel))
        self.loglevel = loglevel

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
