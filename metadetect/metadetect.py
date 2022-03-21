import logging
import time

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


def do_metadetect(config, mbobs, rng, shear_band_combs=None):
    """Run metadetect on the multi-band observations.

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
    shear_band_combs: list of list of int, optional
        If given, each element of the outer list is a list of indices into mbobs to use
        for shear measurement. Shear measurements will be made for each element of the
        outer list. If None, then shear measurements will be made for all entries in
        mbobs.

    Returns
    -------
    res: dict
        The fitting data keyed on the shear component.
    """
    md = Metadetect(
        config, mbobs, rng,
        shear_band_combs=shear_band_combs,
    )
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

    ksigma - Perform a pre-PSF weighted moments measurement. The shear measurement
           is a moment computed from the inverse variance weighted average across
           the bands.

    pgauss - Perform a pre-PSF weighted moments measurement with a Gaussian filter.
             The shear measurement is a moment computed from the inverse variance
             weighted average across the bands.

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
    shear_band_combs: list of list of int, optional
        If given, each element of the outer list is a list of indices into mbobs to use
        for shear measurement. Shear measurements will be made for each element of the
        outer list. If None, then shear measurements will be made for all entries in
        mbobs.
    """
    def __init__(
        self, config, mbobs, rng, show=False,
        shear_band_combs=None,
    ):
        self._show = show

        self._set_config(config)
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.rng = rng

        self._set_fitter()

        # fit all PSFs
        try:
            fitting.fit_all_psfs(self.mbobs, self.rng)
            self._psf_fit_flags = 0
        except BootPSFFailure:
            self._psf_fit_flags = procflags.PSF_FAILURE

        if shear_band_combs is None:
            shear_band_combs = [
                list(range(self.nband)),
            ]

        self._shear_band_combs = shear_band_combs

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

    def _get_ormask_and_bmask(self, mbobs):
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

    def _get_mfrac(self, mbobs):
        """
        set the masked fraction image, averaged over all bands
        """
        wgts = []
        mfrac = np.zeros_like(mbobs[0][0].image)
        for band, obslist in enumerate(mbobs):
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

        return mfrac

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
        """Run metadetect and set the result."""

        mfrac = self._get_mfrac(self.mbobs)
        any_all_zero_weight = False
        any_all_masked = False
        for obsl in self.mbobs:
            for obs in obsl:
                if np.all(obs.weight == 0):
                    any_all_zero_weight = True

                if np.all((obs.bmask & self['nodet_flags']) != 0):
                    any_all_masked = True

        if not np.any(mfrac < 1) or any_all_zero_weight or any_all_masked:
            self._result = None
            return

        # we do metacal on everything so we can get fluxes for non-shear bands later
        t0 = time.time()
        try:
            mcal_res = self._get_all_metacal(self.mbobs)
        except BootPSFFailure:
            mcal_res = None
        logger.info("metacal took %s seconds", time.time() - t0)

        if mcal_res is None:
            self._result = None
            return

        all_res = {}
        for shear_bands in self._shear_band_combs:
            res = self._go_bands(shear_bands, mcal_res)
            if res is not None:
                for k, v in res.items():
                    if v is None:
                        continue
                    if k not in all_res:
                        all_res[k] = []
                    all_res[k].append(v)

        if len(all_res) == 0:
            all_res = None
        else:
            for k in all_res:
                all_res[k] = np.hstack(all_res[k])

        self._result = all_res

    def _go_bands(self, shear_bands, mcal_res):
        # the flagging, mfrac and psf stats are done only with bands for shear
        _mbobs = ngmix.MultiBandObsList()
        for band in shear_bands:
            _mbobs.append(self.mbobs[band])
        mfrac = self._get_mfrac(_mbobs)
        ormask, bmask = self._get_ormask_and_bmask(_mbobs)
        psf_stats = _get_psf_stats(_mbobs, self._psf_fit_flags)

        _result = {}
        for shear_str, shear_mbobs in mcal_res.items():
            _result[shear_str] = self._measure(
                mbobs=shear_mbobs,
                shear_str=shear_str,
                shear_bands=shear_bands,
                mfrac=mfrac,
                bmask=bmask,
                ormask=ormask,
                psf_stats=psf_stats,
            )

        if len(_result) == 0:
            return None
        else:
            return _result

    def _measure(
        self, *, mbobs, shear_str, shear_bands, mfrac, bmask, ormask, psf_stats
    ):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements.

        we only detect on the shear bands in mbobs.

        we then do flux measurements on the nonshear_mbobs as well if it is given.
        """
        t0 = time.time()
        medsifier = self._do_detect(mbobs, shear_bands)
        if self._show:
            import descwl_coadd.vis
            descwl_coadd.vis.show_image(medsifier.seg)

        all_medsifier = detect.CatalogMEDSifier(
            mbobs,
            medsifier.cat['x'],
            medsifier.cat['y'],
            medsifier.cat['box_size'],
        )
        mbm = all_medsifier.get_multiband_meds()
        mbobs_list = mbm.get_mbobs_list()
        logger.info("detect took %s seconds", time.time() - t0)

        t0 = time.time()
        res = fit_mbobs_list_wavg(
            mbobs_list=mbobs_list,
            fitter=self._fitter,
            shear_bands=shear_bands,
            bmask_flags=self.get("bmask_flags", 0),
        )

        if res is not None:
            res = self._add_positions_and_psf(
                cat=medsifier.cat,
                res=res,
                shear_str=shear_str,
                mfrac=mfrac,
                bmask=bmask,
                ormask=ormask,
                psf_stats=psf_stats,
            )
        logger.info("src measurements took %s seconds", time.time() - t0)

        return res

    def _add_positions_and_psf(
        self, *, cat, res, shear_str, mfrac, bmask, ormask, psf_stats
    ):
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
            ('ormask_noshear', 'i4'),
            ('mfrac_noshear', 'f4'),
            ('bmask_noshear', 'i4'),
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

        newres['psfrec_flags'][:] = psf_stats['flags']
        newres['psfrec_g'][:, 0] = psf_stats['g1']
        newres['psfrec_g'][:, 1] = psf_stats['g2']
        newres['psfrec_T'][:] = psf_stats['T']

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
                rows=newres['sx_row'],
                cols=newres['sx_col'],
                mask=ormask,
            )
            newres["ormask_noshear"] = _fill_in_mask_col(
                mask_region=ormask_region,
                rows=newres['sx_row_noshear'],
                cols=newres['sx_col_noshear'],
                mask=ormask,
            )

            newres["bmask"] = _fill_in_mask_col(
                mask_region=bmask_region,
                rows=newres['sx_row'],
                cols=newres['sx_col'],
                mask=bmask,
            )
            newres["bmask_noshear"] = _fill_in_mask_col(
                mask_region=bmask_region,
                rows=newres['sx_row_noshear'],
                cols=newres['sx_col_noshear'],
                mask=bmask,
            )

            if np.any(mfrac > 0):
                newres["mfrac"] = measure_mfrac(
                    mfrac=mfrac,
                    x=newres["sx_col"],
                    y=newres["sx_row"],
                    box_sizes=cat["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                )

                newres["mfrac_noshear"] = measure_mfrac(
                    mfrac=mfrac,
                    x=newres["sx_col_noshear"],
                    y=newres["sx_row_noshear"],
                    box_sizes=cat["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                )
            else:
                newres["mfrac"] = 0
                newres["mfrac_noshear"] = 0

        return newres

    def _do_detect(self, mbobs, shear_bands):
        """
        use a MEDSifier to run detection
        """
        shear_mbobs = ngmix.MultiBandObsList()
        for band in shear_bands:
            shear_mbobs.append(mbobs[band])

        return detect.MEDSifier(
            mbobs=shear_mbobs,
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


def _get_psf_stats(mbobs, global_flags):
    if global_flags != 0:
        flags = procflags.PSF_FAILURE
        g1 = np.nan
        g2 = np.nan
        T = np.nan
    else:
        try:
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
                raise BootPSFFailure(
                    'zero weights, could not get mean psf properties'
                )
            g1 = g1sum/wsum
            g2 = g2sum/wsum
            T = Tsum/wsum

            flags = 0

        except BootPSFFailure:
            flags = procflags.PSF_FAILURE
            g1 = np.nan
            g2 = np.nan
            T = np.nan

    return {
        'flags': flags,
        'g1': g1,
        'g2': g2,
        'T': T,
    }


def _get_bmask_ormask(mbobs):
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

    return bmask, ormask


def _get_mfrac(mbobs):
    """
    set the masked fraction image, averaged over all bands
    """
    wgts = []
    mfrac = np.zeros_like(mbobs[0][0].image)
    for band, obslist in enumerate(mbobs):
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

    return mfrac


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
