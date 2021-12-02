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
from .fitting import fit_mbobs_list_wavg

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

    This class runs the metadetect shear fitting algorithm for observations
    consisting of one image per band. It can use several different fitters for the
    actual shape measurement.

    wmom - Perform a post-PSF weighted moments measurement. The shear measurement
           is a moment computed from the inverse variance weighted average across
           the bands.

    gauss - Perform a joint fit of a Gaussian across the bands.

    ksigma - Perform a pre-PSF weighted moments measurement. The shear measurement
           is a moment computed from the inverse variance weighted average across
           the bands.

    If `nonshear_mbobs` is given, then metacal images for these additional observations
    are made, but only used to get a flux measurement. For the different fitting
    options, this works slightly differently.

    wmom - Perform a post-PSF weighted flux measurement.

    gauss - Peform a second joint fit across all bands used for shear and the ones
            non-shear bands, keeping only the fluxes. We repeat the measurements
            on the bands used for shear so that colors from the flux measurements
            have the same effective aperture.

    ksigma - Perform a pre-PSF weighted flux measurement.

    Parameters
    ----------
    config: dict
        Configuration dictionary. Possible entries are

            metacal
            weight
            model

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
        self['nodet_flags'] = self.get('nodet_flags', 0)

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
            msk = obs.weight > 0
            if not np.any(msk):
                wgt = 0
            else:
                wgt = np.median(obs.weight[msk])
            if hasattr(obs, "mfrac"):
                mfrac += (obs.mfrac * wgt)
            wgts.append(wgt)

        if np.sum(wgts) > 0:
            mfrac = mfrac / np.sum(wgts)
        else:
            mfrac[:, :] = 1.0

        self.mfrac = mfrac

    def _set_fitter(self):
        """
        set the fitter to be used
        """
        self['model'] = self.get('model', 'wmom')

        if self['model'] == 'wmom':
            self._fitter = ngmix.gaussmom.GaussMom(fwhm=self["weight"]["fwhm"])
        elif self['model'] == 'ksigma':
            self._fitter = ngmix.prepsfmom.KSigmaMom(fwhm=self["weight"]["fwhm"])
        elif self['model'] == "pgauss":
            self._fitter = ngmix.prepsfmom.PGaussMom(fwhm=self["weight"]["fwhm"])
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
        any_all_zero_weight = False
        any_all_masked = False
        for obsl in self.mbobs:
            for obs in obsl:
                if np.all(obs.weight == 0):
                    any_all_zero_weight = True

                if np.all((obs.bmask & self['nodet_flags']) != 0):
                    any_all_masked = True

        if not np.any(self.mfrac < 1) or any_all_zero_weight or any_all_masked:
            self._result = None
            return

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

        we only detect on the shear bands in mbobs.

        we then do flux measurements on the nonshear_mbobs as well if it is given.
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

        res = fit_mbobs_list_wavg(
            mbobs_list=mbobs_list,
            fitter=self._fitter,
            nonshear_mbobs_list=nonshear_mbobs_list,
            bmask_flags=self.get("bmask_flags", 0),
        )

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
            ('bmask', 'i4'),
            ('ormask_det', 'i4'),
            ('mfrac_det', 'f4'),
            ('bmask_det', 'i4'),
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

            rows_noshear, cols_noshear = shearpos.unshear_positions_obs(
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

            newres["ormask"] = _fill_in_mask_col(
                mask_region=ormask_region,
                rows=newres['sx_row_noshear'],
                cols=newres['sx_col_noshear'],
                mask=self.ormask,
            )
            newres["ormask_det"] = _fill_in_mask_col(
                mask_region=ormask_region,
                rows=newres['sx_row'],
                cols=newres['sx_col'],
                mask=self.ormask,
            )

            newres["bmask"] = _fill_in_mask_col(
                mask_region=bmask_region,
                rows=newres['sx_row_noshear'],
                cols=newres['sx_col_noshear'],
                mask=self.bmask,
            )
            newres["bmask_det"] = _fill_in_mask_col(
                mask_region=bmask_region,
                rows=newres['sx_row'],
                cols=newres['sx_col'],
                mask=self.bmask,
            )

            if np.any(self.mfrac > 0):
                newres["mfrac"] = measure_mfrac(
                    mfrac=self.mfrac,
                    x=newres["sx_col_noshear"],
                    y=newres["sx_row_noshear"],
                    box_sizes=cat["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                )

                newres["mfrac_det"] = measure_mfrac(
                    mfrac=self.mfrac,
                    x=newres["sx_col"],
                    y=newres["sx_row"],
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
            nodet_flags=self['nodet_flags'],
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
            fitting.fit_all_psfs(self.mbobs, self.rng)

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
            g1 = np.nan
            g2 = np.nan
            T = np.nan

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


def _fill_in_mask_col(*, mask_region, rows, cols, mask):
    dims = mask.shape
    rclip = _clip_and_round(rows, dims[0])
    cclip = _clip_and_round(cols, dims[1])

    if mask_region > 1:
        res = np.zeros_like(rows, dtype=np.int32)
        for ind in range(rows.size):
            lr = int(min(
                dims[0]-1,
                max(0, rclip[ind] - mask_region)))
            ur = int(min(
                dims[0]-1,
                max(0, rclip[ind] + mask_region)))

            lc = int(min(
                dims[1]-1,
                max(0, cclip[ind] - mask_region)))
            uc = int(min(
                dims[1]-1,
                max(0, cclip[ind] + mask_region)))

            res[ind] = np.bitwise_or.reduce(
                mask[lr:ur+1, lc:uc+1],
                axis=None,
            )
    else:
        res = mask[rclip, cclip].astype(np.int32)

    return res
