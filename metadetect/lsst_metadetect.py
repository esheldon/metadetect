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
                            afw_image.ImageD(psf_image.astype(np.float))
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
            res = self._add_positions_and_psf(medsifier, res, shear_str)

        return res

    def _add_positions_and_psf(self, medsifier, res, shear_str):
        """
        TODO add positionis etc.
        """

        new_dt = [
            ('row', 'f4'),
            ('col', 'f4'),
            ('row_noshear', 'f4'),
            ('col_noshear', 'f4'),
            ('ormask', 'i4'),
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

        newres['psfrec_flags'][:] = self.psf_stats['flags']
        newres['psfrec_g'][:, 0] = self.psf_stats['g1']
        newres['psfrec_g'][:, 1] = self.psf_stats['g2']
        newres['psfrec_T'][:] = self.psf_stats['T']

        sources = medsifier.sources
        det_exp = medsifier.det_exp

        for i, rec in enumerate(sources):
            orig_cen = det_exp.getWcs().skyToPixel(rec.getCoord())
            if np.isnan(orig_cen.getY()):
                self.log.debug('falling back on integer location')
                # fall back to integer pixel location
                peak = rec.getFootprint().getPeaks()[0]
                orig_cen = peak.getI()

            newres['row'][i] = orig_cen.getY()
            newres['col'][i] = orig_cen.getX()
            # print(newres['row'][i], newres['col'][i])
        '''
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
        '''
        return newres

    def _set_ormask(self):
        """
        fake ormask for now
        """
        self.ormask = np.zeros(self.mbobs[0][0].image.shape, dtype='i4')

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
            res = self._add_original_psf(res, shear_str)

        return res

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
