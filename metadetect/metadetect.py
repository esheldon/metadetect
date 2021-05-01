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


def do_metadetect(config, mbobs, rng, nonshear_mbobs=None):
    """
    meta-detect on the multi-band observations.  For parameters, see
    docs on the Metadetect class

    Parameters
    ----------
    config: dict
        Configuration dictionary. Possible entries are

            metacal
            weight
            model
            flux

    mbobs: ngmix.MultiBandObsList
        We will do detection and measurements on these images
    rng: numpy.random.RandomState
        Random number generator
    nonshear_mbobs: ngmix.MultiBandObsList, optional
        If not None and 'flux' is given in the config, then this mbobs will
        be sheared and have flux measurements made at the detected positions in
        the `mbobs`.

    Returns
    -------
    res: dict
        The fitting data keyed on the shear component.
    """
    md = Metadetect(config, mbobs, rng, nonshear_mbobs=nonshear_mbobs)
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
            weight
            model
            flux

    mbobs: ngmix.MultiBandObsList
        We will do detection and measurements on these images
    rng: numpy.random.RandomState
        Random number generator
    show: bool, optional
        If True, show the images using descwl_coadd.vis.
    nonshear_mbobs: ngmix.MultiBandObsList, optional
        If not None and 'flux' is given in the config, then this mbobs will
        be sheared and have flux measurements made at the detected positions in
        the `mbobs`.
    """
    def __init__(self, config, mbobs, rng, show=False, nonshear_mbobs=None):

        self._show = show

        self._set_config(config)
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.rng = rng

        self.nonshear_mbobs = nonshear_mbobs
        self.nonshear_nband = (
            len(nonshear_mbobs)
            if nonshear_mbobs is not None
            else 0
        )

        if 'psf' in config:
            self._fit_original_psfs()

        self._set_fitter()

        self._set_ormask_and_bmask()
        self._set_mfrac()

        if "flux" in self:
            self._set_flux_fitter()

    def _set_ormask_and_bmask(self):
        """
        set the ormask and bmask, ored from all epochs
        """

        for band, obslist in enumerate(self.mbobs):
            nepoch = len(obslist)
            assert nepoch == 1, 'expected 1 epoch, got %d' % nepoch

            obs = obslist[0]

            if band == 0:
                ormask = obs.ormask.copy()
                bmask = obs.bmask.copy()
            else:
                ormask |= obs.ormask
                bmask |= obs.bmask

        self.ormask = ormask
        self.bmask = bmask

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
            odict = self._get_all_metacal(self.mbobs)
        except BootPSFFailure:
            odict = None

        if 'flux' in self and self.nonshear_nband > 0:
            try:
                nonshear_odict = self._get_all_metacal(self.nonshear_mbobs)
            except BootPSFFailure:
                nonshear_odict = None

        if (
            odict is None
            or (
                'flux' in self
                and self.nonshear_nband > 0
                and nonshear_odict is None
            )
        ):
            self._result = None
        else:
            self._result = {}
            for shear_str, mbobs in odict.items():
                if 'flux' in self and self.nonshear_nband > 0:
                    nonshear_mbobs = nonshear_odict[shear_str]
                else:
                    nonshear_mbobs = None
                self._result[shear_str] = self._measure(
                    mbobs, shear_str, nonshear_mbobs=nonshear_mbobs
                )

    def _set_fitter(self):
        """
        set the fitter to be used
        """
        self['model'] = self.get('model', 'wmom')

        if self['model'] == 'wmom':
            self._fitter = fitting.Moments(self, self.rng)
        elif self['model'] == 'gauss':
            if ngmix.__version__[0:2] == 'v1':
                self._fitter = fitting.MaxLikeNgmixv1(
                    self, self.rng, self.nband,
                )
            else:
                self._fitter = fitting.MaxLike(self, self.rng, self.nband)
        else:
            raise ValueError("bad model: '%s'" % self['model'])

    def _set_flux_fitter(self):
        """
        set the fitter to be used
        """
        self['flux']['model'] = self['flux'].get('model', 'wmom')

        if self['flux']['model'] == 'wmom':
            self['flux']['bmask_flags'] = self['bmask_flags']
            self._flux_fitter = fitting.Moments(self['flux'], self.rng)
        else:
            raise ValueError("bad model: '%s'" % self['flux']['model'])

    def _measure(self, mbobs, shear_str, nonshear_mbobs=None):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements.

        flux measurements will be made on the nonshear_mbobs as well
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

            if 'flux' in self:
                res = self._run_flux_fitter(
                    res,
                    medsifier.cat,
                    mbobs,
                    nonshear_mbobs=nonshear_mbobs,
                )

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
            ('bmask', 'i4'),
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
                ormask_region = self['ormask_region']
            elif 'mask_region' in self and self['mask_region'] > 1:
                ormask_region = self['mask_region']
            else:
                ormask_region = 1

            if 'mask_region' in self and self['mask_region'] > 1:
                bmask_region = self['mask_region']
            else:
                bmask_region = 1

                logger.debug(
                    'ormask|bmask region: %s|%s',
                    ormask_region,
                    bmask_region,
                )

            if ormask_region > 1:
                for ind in range(cat.size):
                    lr = int(min(
                        dims[0]-1,
                        max(0, rclip[ind] - ormask_region)))
                    ur = int(min(
                        dims[0]-1,
                        max(0, rclip[ind] + ormask_region)))

                    lc = int(min(
                        dims[1]-1,
                        max(0, cclip[ind] - ormask_region)))
                    uc = int(min(
                        dims[1]-1,
                        max(0, cclip[ind] + ormask_region)))

                    newres['ormask'][ind] = np.bitwise_or.reduce(
                        self.ormask[lr:ur+1, lc:uc+1],
                        axis=None,
                    )
            else:
                newres['ormask'] = self.ormask[rclip, cclip]

            if bmask_region > 1:
                for ind in range(cat.size):
                    lr = int(min(
                        dims[0]-1,
                        max(0, rclip[ind] - bmask_region)))
                    ur = int(min(
                        dims[0]-1,
                        max(0, rclip[ind] + bmask_region)))

                    lc = int(min(
                        dims[1]-1,
                        max(0, cclip[ind] - bmask_region)))
                    uc = int(min(
                        dims[1]-1,
                        max(0, cclip[ind] + bmask_region)))

                    newres['bmask'][ind] = np.bitwise_or.reduce(
                        self.bmask[lr:ur+1, lc:uc+1],
                        axis=None,
                    )
            else:
                newres['bmask'] = self.bmask[rclip, cclip]

            if np.any(self.mfrac > 0):
                # we are using the positions with the metacal shear removed
                # for this.
                newres["mfrac"] = measure_mfrac(
                    mfrac=self.mfrac,
                    x=newres["sx_col_noshear"],
                    y=newres["sx_row_noshear"],
                    box_sizes=cat["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                )
            else:
                newres["mfrac"] = 0

        return newres

    def _run_flux_fitter(
        self,
        res,
        cat,
        mbobs,
        nonshear_mbobs=None,
    ):
        names = ["shear%d" % i for i in range(self.nband)]
        fit_mbobs = ngmix.MultiBandObsList()

        if nonshear_mbobs is not None:
            names += ["nonshear%d" % i for i in range(self.nonshear_nband)]
            for obsl in self.nonshear_mbobs:
                fit_mbobs.append(obsl)

        medsifier = detect.CatalogMEDSifier(
            fit_mbobs, cat['x'], cat['y'], cat['box_size']
        )
        mbm = medsifier.get_multiband_meds()
        mbobs_list = mbm.get_mbobs_list()

        if self['flux']['model'] in ['wmom']:
            res = self._run_flux_fitter_mbobs_sep(
                res, mbobs_list, names, self['flux']['model']
            )
        else:
            raise RuntimeError(
                "flux model %s not supported!" % self['flux']['model']
            )
        return res

    def _run_flux_fitter_mbobs_sep(
        self,
        res,
        mbobs_list,
        names,
        model,
    ):
        add_dt = []
        for name in names:
            nm_mod = name + "_" + model
            add_dt += [
                (nm_mod + "_flux_flags", 'f8'),
                (nm_mod + "_flux", 'f8'),
                (nm_mod + "_flux_err", 'f8'),
            ]
        newres = eu.numpy_util.add_fields(
            res,
            add_dt,
        )

        for i, mbobs in enumerate(mbobs_list):
            for name, obsl in zip(names, mbobs):
                nm_mod = name + "_" + model
                band_mbobs = ngmix.MultiBandObsList()
                band_mbobs.append(obsl)
                band_res = self._flux_fitter.go([band_mbobs])
                newres[nm_mod + "_flux_flags"][i] = band_res["flags"][0]
                newres[nm_mod + "_flux"][i] = band_res[model + "_flux"][0]
                newres[nm_mod + "_flux_err"][i] = band_res[model + "_flux_err"][0]
                newres["flags"] |= band_res["flags"][0]

        return newres

    def _do_detect(self, mbobs):
        """
        use a MEDSifier to run detection
        """
        return detect.MEDSifier(
            mbobs=mbobs,
            sx_config=self['sx'],
            meds_config=self['meds'],
            maskflags=self['maskflags'],
        )

    def _get_all_metacal(self, mbobs):
        """
        get the sheared versions of the observations
        """

        odict = ngmix.metacal.get_all_metacal(
            mbobs,
            rng=self.rng,
            **self['metacal']
        )

        if self._show:
            import descwl_coadd.vis

            orig_mbobs = mbobs

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
        self['maskflags'] = self.get('maskflags', 0)

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
