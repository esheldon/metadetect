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
        self._mbobs = mbobs
        self._nband = len(mbobs)
        self._rng = rng

        # set the config
        self.update(config)
        assert 'metacal' in self, \
            'metacal setting must be present in config'
        assert 'sx' in self, \
            'sx setting must be present in config'
        assert 'meds' in self, \
            'meds setting must be present in config'
        self['nodet_flags'] = self.get('nodet_flags', 0)
        self["bmask_flags"] = self.get("bmask_flags", 0)
        if 'ormask_region' in config:
            raise RuntimeError("ormask_region is not supported, use mask_region!")
        if 'bmask_region' in config:
            raise RuntimeError("bmask_region is not supported, use mask_region!")

        # set the fitter
        self['model'] = self.get('model', 'wmom')
        if self['model'] == 'wmom':
            self._fitter = ngmix.gaussmom.GaussMom(fwhm=self["weight"]["fwhm"])
        elif self['model'] == 'ksigma':
            self._fitter = ngmix.prepsfmom.KSigmaMom(fwhm=self["weight"]["fwhm"])
        elif self['model'] == "pgauss":
            self._fitter = ngmix.prepsfmom.PGaussMom(fwhm=self["weight"]["fwhm"])
        else:
            raise ValueError("bad model: '%s'" % self['model'])

        # fit all PSFs
        try:
            fitting.fit_all_psfs(self._mbobs, self._rng)
            self._psf_fit_flags = 0
        except BootPSFFailure:
            self._psf_fit_flags = procflags.PSF_FAILURE

        if shear_band_combs is None:
            shear_band_combs = [
                list(range(self._nband)),
            ]

        self._shear_band_combs = shear_band_combs

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

        mfrac = _get_mfrac(self._mbobs)
        any_all_zero_weight = False
        any_all_masked = False
        for obsl in self._mbobs:
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
            mcal_res = self._get_all_metacal(self._mbobs)
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
        # the flagging and mfrac is done only with bands for shear
        mbobs = ngmix.MultiBandObsList()
        for band in shear_bands:
            mbobs.append(self._mbobs[band])
        mfrac = _get_mfrac(mbobs)
        bmask, ormask = _get_bmask_ormask(mbobs)

        # psf stats come from the original mbobs
        psf_stats = _get_psf_stats(self._mbobs, shear_bands, self._psf_fit_flags)

        _result = {}
        for shear_str, shear_mbobs in mcal_res.items():
            _result[shear_str] = self._measure_mdet(
                mbobs=shear_mbobs,
                shear_str=shear_str,
                shear_bands=shear_bands,
                mfrac=mfrac,
                bmask=bmask,
                ormask=ormask,
                fitter=self._fitter,
                psf_stats=psf_stats,
            )

        if len(_result) == 0:
            return None
        else:
            return _result

    def _measure_mdet(
        self, *, mbobs, shear_str, shear_bands, mfrac, bmask, ormask, fitter,
        psf_stats,
    ):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements.

        we only detect on the shear bands in mbobs.

        we then do flux measurements on everything
        """
        cat, mbobs_list = self._do_detection(mbobs, shear_bands)

        t0 = time.time()
        res = fit_mbobs_list_wavg(
            mbobs_list=mbobs_list,
            fitter=fitter,
            bmask_flags=self["bmask_flags"],
            shear_bands=shear_bands,
        )
        logger.info("src measurements took %s seconds", time.time() - t0)

        if res is not None:
            res = _add_positions(
                cat=cat,
                res=res,
                shear_str=shear_str,
                obs=mbobs[0][0],
                metacal_step=self['metacal'].get("step", shearpos.DEFAULT_STEP),
            )
            res = _add_psf_stats(
                res=res,
                psf_stats=psf_stats,
            )
            res = _add_bmask_and_ormask(
                res=res, bmask=bmask, ormask=ormask,
                mask_region=self.get("mask_region", 1),
            )
            res = _add_mfrac(
                res=res, mfrac=mfrac,
                mfrac_fwhm=self.get("mfrac_fwhm", None),
                box_sizes=cat["box_size"],
                obs=mbobs[0][0],
            )

        return res

    def _get_all_metacal(self, mbobs):
        """
        get the sheared versions of the observations
        """

        odict = ngmix.metacal.get_all_metacal(
            mbobs,
            rng=self._rng,
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

    def _do_detection(self, mbobs, shear_bands):
        shear_mbobs = ngmix.MultiBandObsList()
        for band in shear_bands:
            shear_mbobs.append(mbobs[band])
        t0 = time.time()
        medsifier = detect.MEDSifier(
            mbobs=shear_mbobs,
            sx_config=self['sx'],
            meds_config=self['meds'],
            nodet_flags=self['nodet_flags'],
        )
        logger.info("detect took %s seconds", time.time() - t0)

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

        return medsifier.cat, mbobs_list


def _add_positions(
    *, cat, res, shear_str, obs, metacal_step,
):
    new_dt = [
        ('sx_row', 'f4'),
        ('sx_col', 'f4'),
        ('sx_row_noshear', 'f4'),
        ('sx_col_noshear', 'f4'),
    ]
    newres = eu.numpy_util.add_fields(
        res,
        new_dt,
    )

    if res.size > 0:
        newres['sx_col'] = cat['x']
        newres['sx_row'] = cat['y']

        rows_noshear, cols_noshear = shearpos.unshear_positions_obs(
            newres['sx_row'],
            newres['sx_col'],
            shear_str,
            obs,  # an example for jacobian and image shape
            # default is 0.01 but make sure to use the passed in default
            # if needed
            step=metacal_step,
        )

        newres['sx_row_noshear'] = rows_noshear
        newres['sx_col_noshear'] = cols_noshear

    return newres


def _add_mfrac(
    *, res, mfrac, mfrac_fwhm, box_sizes, obs
):
    new_dt = [
        ('mfrac', 'f4'),
        ('mfrac_noshear', 'f4'),
    ]
    newres = eu.numpy_util.add_fields(
        res,
        new_dt,
    )

    if res.size > 0:
        if np.any(mfrac > 0):
            newres["mfrac"] = measure_mfrac(
                mfrac=mfrac,
                x=newres["sx_col"],
                y=newres["sx_row"],
                box_sizes=box_sizes,
                obs=obs,
                fwhm=mfrac_fwhm,
            )

            newres["mfrac_noshear"] = measure_mfrac(
                mfrac=mfrac,
                x=newres["sx_col_noshear"],
                y=newres["sx_row_noshear"],
                box_sizes=box_sizes,
                obs=obs,
                fwhm=mfrac_fwhm,
            )
        else:
            newres["mfrac"] = 0
            newres["mfrac_noshear"] = 0

    return newres


def _add_bmask_and_ormask(*, res, bmask, ormask, mask_region):
    new_dt = [
        ('ormask', 'i4'),
        ('bmask', 'i4'),
        ('ormask_noshear', 'i4'),
        ('bmask_noshear', 'i4'),
    ]
    newres = eu.numpy_util.add_fields(
        res,
        new_dt,
    )

    logger.debug('ormask|bmask region: %s', mask_region)

    if res.size > 0:
        newres["ormask"] = _fill_in_mask_col(
            mask_region=mask_region,
            rows=newres['sx_row'],
            cols=newres['sx_col'],
            mask=ormask,
        )
        newres["ormask_noshear"] = _fill_in_mask_col(
            mask_region=mask_region,
            rows=newres['sx_row_noshear'],
            cols=newres['sx_col_noshear'],
            mask=ormask,
        )

        newres["bmask"] = _fill_in_mask_col(
            mask_region=mask_region,
            rows=newres['sx_row'],
            cols=newres['sx_col'],
            mask=bmask,
        )
        newres["bmask_noshear"] = _fill_in_mask_col(
            mask_region=mask_region,
            rows=newres['sx_row_noshear'],
            cols=newres['sx_col_noshear'],
            mask=bmask,
        )

    return newres


def _add_psf_stats(res, psf_stats):
    new_dt = [
        ('psfrec_flags', 'i4'),  # psfrec is the original psf
        ('psfrec_g', 'f8', 2),
        ('psfrec_T', 'f8'),
    ]
    newres = eu.numpy_util.add_fields(
        res,
        new_dt,
    )

    if res.size > 0:
        newres['psfrec_flags'] = psf_stats['flags']
        newres['psfrec_g'][:, 0] = psf_stats['g1']
        newres['psfrec_g'][:, 1] = psf_stats['g2']
        newres['psfrec_T'] = psf_stats['T']

    return newres


def _get_psf_stats(mbobs, shear_bands, global_flags):
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

            for band in shear_bands:
                obslist = mbobs[band]
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
