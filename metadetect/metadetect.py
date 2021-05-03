import logging
import copy

import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure
import esutil as eu

from .util import Namer
from . import detect
from . import fitting
from . import procflags
from . import shearpos
from .mfrac import measure_mfrac

logger = logging.getLogger(__name__)


def do_metadetect(config, mbobs, rng, nonshear_mbobs=None):
    """Run metadetect on the multi-band observations.

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
    """Metadetect fitter for multi-band observations.

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
        If not None athen this mbobs will be sheared and have flux measurements
        made at the detected positions in the `mbobs`.
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

    def _set_fitter(self):
        """
        set the fitter to be used
        """
        self['model'] = self.get('model', 'wmom')

        if self['model'] == 'wmom':
            self._fitter = fitting.Moments(
                self, self.rng, keep_flux=True,
            )
            if self.nonshear_mbobs is not None:
                self._nonshear_fitter = fitting.Moments(
                    self, self.rng, keep_flux=True
                )
        elif self['model'] == 'gauss':
            if ngmix.__version__[0:2] == 'v1':
                self._fitter = fitting.MaxLikeNgmixv1(
                    self, self.rng, self.nband, keep_flux=True
                )
                if self.nonshear_mbobs is not None:
                    self._nonshear_fitter = fitting.MaxLikeNgmixv1(
                        self, self.rng, self.nonshear_nband,
                        keep_flux=True,
                    )
            else:
                self._fitter = fitting.MaxLike(
                    self, self.rng, self.nband, keep_flux=True,
                )
                if self.nonshear_mbobs is not None:
                    self._nonshear_fitter = fitting.MaxLike(
                        self, self.rng, self.nonshear_nband,
                        keep_flux=True,
                    )
        else:
            raise ValueError("bad model: '%s'" % self['model'])

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

        if self.nonshear_mbobs is not None:
            try:
                nonshear_odict = self._get_all_metacal(self.nonshear_mbobs)
            except BootPSFFailure:
                nonshear_odict = None

        if (
            odict is None
            or (
                self.nonshear_mbobs is not None
                and nonshear_odict is None
            )
        ):
            self._result = None
        else:
            self._result = {}
            for shear_str, mbobs in odict.items():
                if self.nonshear_mbobs is not None and nonshear_odict is not None:
                    nonshear_mbobs = nonshear_odict[shear_str]
                else:
                    nonshear_mbobs = None
                self._result[shear_str] = self._measure(
                    mbobs, shear_str, nonshear_mbobs=nonshear_mbobs
                )

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

        if nonshear_mbobs is not None:
            nonshear_medsifier = detect.CatalogMEDSifier(
                nonshear_mbobs,
                medsifier.cat['x'],
                medsifier.cat['y'],
                medsifier.cat['box_size'],
            )
            nonshear_mbm = nonshear_medsifier.get_multiband_meds()
            nonshear_mbobs_list = nonshear_mbm.get_mbobs_list()
        else:
            nonshear_mbobs_list = None

        res = self._run_fitter(mbobs_list, nonshear_mbobs_list=nonshear_mbobs_list)

        if res is not None:
            res = self._add_positions_and_psf(medsifier.cat, res, shear_str)

        return res

    def _run_fitter(self, mbobs_list, nonshear_mbobs_list=None):
        if self['model'] in ['wmom']:
            return self._run_fitter_mbobs_sep(
                mbobs_list, nonshear_mbobs_list=nonshear_mbobs_list,
            )
        elif self['model'] in ['gauss']:
            return self._run_fitter_mbobs_comb(
                mbobs_list, nonshear_mbobs_list=nonshear_mbobs_list,
            )
        else:
            raise RuntimeError(
                "shear model %s not supported!" % self['model']
            )

    def _run_fitter_mbobs_comb(self, mbobs_list, nonshear_mbobs_list=None):
        res = self._fitter.go(mbobs_list)
        if nonshear_mbobs_list is not None:
            res_nonshear = self._fitter.go(nonshear_mbobs_list)
            tot_nband = self.nband + self.nonshear_nband
        else:
            tot_nband = self.nband

        n = Namer(front=self['model'])
        new_dt = [
            (n("band_flux"), "f8", tot_nband),
            (n("band_flux_err"), "f8", tot_nband),
        ]
        newres = eu.numpy_util.add_fields(
            res,
            new_dt,
        )
        newres[n("band_flux")][:, :self.nband] = res[n('flux')]
        newres[n("band_flux_err")][:, :self.nband] = res[n('flux_err')]

        if nonshear_mbobs_list is not None:
            newres[n("band_flux")][:, self.nband:] = res_nonshear[n('flux')]
            newres[n("band_flux_err")][:, self.nband:] = res_nonshear[n('flux_err')]

        # remove the flux column
        new_dt = [dt for dt in newres.dtype.descr if dt[0] not in ["flux", "flux_err"]]
        final_res = np.zeros(newres.shape[0], dtype=new_dt)
        for c in final_res.dtype.names:
            final_res[c] = newres[c]

        return final_res

    def _run_fitter_mbobs_sep(self, mbobs_list, nonshear_mbobs_list=None):
        # run the fits
        band_res = []
        all_is_shear_band = []

        for band in range(self.nband):
            all_is_shear_band.append(True)
            band_mbobs_list = []
            for mbobs in mbobs_list:
                _mbobs = ngmix.MultiBandObsList()
                _mbobs.meta = copy.deepcopy(mbobs.meta)
                _mbobs.append(mbobs[band])
                band_mbobs_list.append(_mbobs)
            band_res.append(self._fitter.go(band_mbobs_list))

        if nonshear_mbobs_list is not None:
            for band in range(self.nonshear_nband):
                all_is_shear_band.append(False)
                band_mbobs_list = []
                for mbobs in nonshear_mbobs_list:
                    _mbobs = ngmix.MultiBandObsList()
                    _mbobs.meta = copy.deepcopy(mbobs.meta)
                    _mbobs.append(mbobs[band])
                    band_mbobs_list.append(_mbobs)
                band_res.append(self._fitter.go(band_mbobs_list))

        # combine the data by inverse variance weighted averages
        tot_nband = self.nband + self.nonshear_nband
        n = Namer(front=self['model'])
        dt = [
            ("flags", 'i4'),
            (n("flags"), 'i4'),
            (n("s2n"), "f8"),
            (n("T"), "f8"),
            (n("T_err"), "f8"),
            (n("g"), "f8", 2),
            (n("g_cov"), "f8", (2, 2)),
            ('psf_g', 'f8', 2),
            ('psf_T', 'f8'),
            (n("T_ratio"), "f8"),
            (n("band_flux"), "f8", tot_nband),
            (n("band_flux_err"), "f8", tot_nband),
        ]
        res = np.zeros(len(mbobs_list), dtype=dt)
        for ind in range(len(mbobs_list)):
            # extract the wgts and band results
            wgts = []
            all_bres = []
            for i, obslist in enumerate(mbobs_list[ind]):
                wgts.append(np.median(obslist[0].weight))
                all_bres.append(band_res[i][ind:ind+1])

            if nonshear_mbobs_list is not None:
                for i in range(self.nband, tot_nband):
                    all_bres.append(band_res[i][ind:ind+1])
                    wgts.append(1)

            wgts = np.array(wgts)
            wgts[0:self.nband] = wgts[0:self.nband] / np.sum(wgts[0:self.nband])

            # compute the averages
            band_flux = []
            band_flux_err = []
            flux = 0.0
            flux_var = 0.0
            for wgt, bres, is_shear_band in zip(wgts, all_bres, all_is_shear_band):
                res['flags'][ind] |= bres['flags'][0]
                res[n('flags')][ind] |= bres['flags'][0]

                if is_shear_band:
                    flux += (wgt * bres[n('flux')][0])
                    flux_var += (wgt * bres[n('flux_err')][0])**2

                    res[n('T')][ind] += (wgt * bres[n('T')][0])
                    res[n('T_err')][ind] += (wgt * bres[n('T_err')][0])**2

                    res[n('g')][ind] += (wgt * bres[n('T')][0] * bres[n('g')][0])
                    res[n('g_cov')][ind] += (
                        bres[n('g_cov')][0]
                        * (wgt * bres[n('T')][0])**2
                    )

                    res['psf_g'][ind] += (wgt * bres['psf_g'][0])
                    res['psf_T'][ind] += (wgt * bres['psf_T'][0])

                band_flux.append(bres[n('flux')][0])
                band_flux_err.append(bres[n('flux_err')][0])

            res[n('g')][ind] = res[n('g')][ind] / res[n('T')][ind]
            res[n('g_cov')][ind] = res[n('g_cov')][ind] / res[n('T')][ind]**2
            res[n('s2n')][ind] = flux / np.sqrt(flux_var)
            res[n('T_err')][ind] = np.sqrt(res[n('T_err')][ind])
            res[n('T_ratio')][ind] = res[n('T')][ind] / res['psf_T'][ind]
            res[n('band_flux')][ind] = band_flux
            res[n('band_flux_err')][ind] = band_flux_err

        if len(res) == 0:
            return None
        else:
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
        if 'psfrec_flags' not in res.dtype.names:
            new_dt += [
                ('psfrec_flags', 'i4'),  # psfrec is the original psf
                ('psfrec_g', 'f8', 2),
                ('psfrec_T', 'f8'),
            ]
        newres = eu.numpy_util.add_fields(
            res,
            new_dt,
        )
        if 'psfrec_flags' not in res.dtype.names:
            newres['psfrec_flags'] = procflags.NO_ATTEMPT

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
