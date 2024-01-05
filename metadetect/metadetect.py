import logging
import time

import numpy as np
import ngmix
from ngmix.gexceptions import BootPSFFailure
import esutil as eu

from . import fitting
from . import procflags
from . import shearpos
from .util import Namer
from .mfrac import measure_mfrac
from .fitting import (
    fit_mbobs_list_wavg,
    combine_fit_res,
    fit_mbobs_list_joint,
    MAX_NUM_SHEAR_BANDS,
)

logger = logging.getLogger(__name__)


def do_metadetect(
    config, mbobs, rng, shear_band_combs=None,
    color_key_func=None, color_dep_mbobs=None,
    det_band_combs=None,
):
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
    det_band_combs: list of list of int or str, optional
        If given, the set of bands to use for detection. The default of None uses all
        of the bands. If the string "shear_bands" is passed, the code uses the bands
        used for shear.
    color_key_func: function, optional
        If given, a function that computes a color or tuple of colors to key the
        `color_dep_mbobs` dictionary given an input set of fluxes from the mbobs.
    color_dep_mbobs: dict of mbobs, optional
        A dictionary of color-dependently rendered observations of the mbobs for use
        in color-dependent metadetect.

    Returns
    -------
    res: dict
        The fitting data keyed on the shear component.
    """
    md = Metadetect(
        config, mbobs, rng,
        shear_band_combs=shear_band_combs,
        color_key_func=color_key_func,
        color_dep_mbobs=color_dep_mbobs,
        det_band_combs=det_band_combs,
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

    am or admom - Use adaptive moments. The shear measurement is compute from fitting
                  adaptive moments on a coadd of the bands used for shear.

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
    det_band_combs: list of list of int or str, optional
        If given, the set of bands to use for detection. The default of None uses all
        of the bands. If the string "shear_bands" is passed, the code uses the bands
        used for shear.
    color_key_func: function, optional
        If given, a function that computes a color or tuple of colors to key the
        `color_dep_mbobs` dictionary given an input set of fluxes from the mbobs. If
        this function returns None, the PSF from the original observation is used.
    color_dep_mbobs: dict of mbobs, optional
        A dictionary of color-dependently rendered observations of the mbobs for use
        in color-dependent metadetect.
    """
    def __init__(
        self, config, mbobs, rng, show=False,
        shear_band_combs=None,
        color_key_func=None,
        color_dep_mbobs=None,
        det_band_combs=None,
    ):
        self._show = show

        self._set_config(config)
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.rng = rng

        self.color_key_func = color_key_func
        self.color_dep_mbobs = color_dep_mbobs
        if (
            (color_dep_mbobs is not None and color_key_func is None)
            or
            (color_dep_mbobs is None and color_key_func is not None)
        ):
            raise RuntimeError(
                "You must both `color_dep_mbobs` and `color_key_func`!"
            )

        self._set_fitter()

        if shear_band_combs is None:
            shear_band_combs = [
                list(range(self.nband)),
            ]

        self._shear_band_combs = shear_band_combs

        if det_band_combs is None:
            det_band_combs = (
                [list(range(self.nband))]
                * len(self._shear_band_combs)
            )
        elif det_band_combs == "shear_bands":
            det_band_combs = self._shear_band_combs

        self._det_band_combs = det_band_combs

    def _set_config(self, config):
        """
        set the config, dealing with defaults
        """

        self.update(config)
        assert 'metacal' in self, \
            'metacal setting must be present in config'
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
        get the masked fraction image, averaged over all bands
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

        def _get_fitter(cfg):
            model = cfg.get('model', 'wmom')
            symmetrize = cfg.get("symmetrize", True)

            if "fwhm_smooth" in cfg.get("weight", {}):
                kwargs = {"fwhm_smooth": cfg["weight"]["fwhm_smooth"]}
            else:
                kwargs = {}

            if model == 'wmom':
                fitter = ngmix.gaussmom.GaussMom(fwhm=cfg["weight"]["fwhm"])
                is_wavg = True
                coadd = False
            elif model == 'ksigma':
                fitter = ngmix.prepsfmom.KSigmaMom(
                    fwhm=cfg["weight"]["fwhm"],
                    **kwargs,
                )
                is_wavg = True
                coadd = False
            elif model == "pgauss":
                fitter = ngmix.prepsfmom.PGaussMom(
                    fwhm=cfg["weight"]["fwhm"],
                    **kwargs,
                )
                is_wavg = True
                coadd = False
            elif model in ["admom", "am", "gauss"]:
                # we pass the name to our codes
                fitter = model
                is_wavg = False

                # we set this defualt
                # it may be used to set the masked fraction measurement
                # aperture
                if "weight" not in cfg:
                    cfg["weight"] = {}
                if "fwhm" not in cfg["weight"]:
                    cfg["weight"]["fwhm"] = 1.2

                coadd = cfg.get("coadd", False)
            else:
                raise ValueError("bad model: '%s'" % model)

            if "fwhm_reg" in cfg.get("weight", {}):
                fwhm_reg = cfg["weight"]["fwhm_reg"]
                fitter.kind = fitter.kind + "_reg%0.2f" % cfg["weight"]["fwhm_reg"]
            else:
                fwhm_reg = 0

            return (
                model, fitter, cfg["weight"]["fwhm"], fwhm_reg,
                is_wavg, symmetrize, coadd,
            )

        if "fitters" in self and (
            "model" in self
            or "weight" in self
            or "symmetrize" in self
            or "coadd" in self
        ):
            raise RuntimeError(
                "You can only specify one of fitters or "
                "model+weight+symmetrize+coadd!"
            )

        if "fitters" in self:
            fitters = []
            fwhms = []
            fwhm_regs = []
            fitter_is_wavg = []
            fitter_symmetrize = []
            fitter_coadd = []
            for fitter_cfg in self["fitters"]:
                _, fitter, fwhm, fwhm_reg, is_wavg, symmetrize, coadd = _get_fitter(
                    fitter_cfg
                )
                fitters.append(fitter)
                fwhms.append(fwhm)
                fwhm_regs.append(fwhm_reg)
                fitter_is_wavg.append(is_wavg)
                fitter_symmetrize.append(symmetrize)
                fitter_coadd.append(coadd)
            self._fitters = fitters
            self._fwhms = fwhms
            self._fwhm_regs = fwhm_regs
            self._fitter_is_wavg = fitter_is_wavg
            self._fitter_symmetrize = fitter_symmetrize
            self._fitter_coadd = fitter_coadd
        else:
            _, fitter, fwhm, fwhm_reg, is_wavg, symmetrize, coadd = _get_fitter(self)
            self._fitters = [fitter]
            self._fwhms = [fwhm]
            self._fwhm_regs = [fwhm_reg]
            self._fitter_is_wavg = [is_wavg]
            self._fitter_symmetrize = [symmetrize]
            self._fitter_coadd = [coadd]

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

        # if the there are no pixels with mfrac < 1 or it is all zero weight
        # or it is all masked, we cannot measure anything so set result to None
        if (not np.any(mfrac < 1)) or any_all_zero_weight or any_all_masked:
            self._result = None
            return

        # we do metacal on everything so we can get fluxes for non-shear bands later
        mcal_res = self._get_mbobs_data(None, list(range(self.nband)))["mcal_res"]

        if mcal_res is None:
            self._result = None
            return

        # past this point, the code should always return a dictionary with the minimal
        # metacal types
        # this indicates that a measurement should have been possible
        # we may find nothing, but that is a different thing
        all_res = {}
        for shear_bands, det_bands in zip(
            self._shear_band_combs, self._det_band_combs
        ):
            if self.color_key_func is not None and self.color_dep_mbobs is not None:
                res = self._go_bands_with_color(shear_bands, mcal_res, det_bands)
            else:
                res = self._go_bands(shear_bands, mcal_res, det_bands)
            if res is not None:
                for k, v in res.items():
                    if v is None:
                        continue
                    if k not in all_res:
                        all_res[k] = []
                    all_res[k].append(v)

        for k in all_res:
            all_res[k] = np.hstack(all_res[k])

        for mcal_type in self['metacal'].get(
            "types", ngmix.metacal.METACAL_MINIMAL_TYPES
        ):
            if mcal_type not in all_res:
                # this can happen if we do not detect sources for one of the
                # metacal images
                all_res[mcal_type] = None

        self._result = all_res

    def _go_bands(self, shear_bands, mcal_res, det_bands):
        kdata = self._get_mbobs_data(None, shear_bands)

        _result = {}
        for shear_str, shear_mbobs in mcal_res.items():
            cat, mbobs_list = self._do_detect(
                shear_mbobs,
                det_bands,
            )
            _result[shear_str] = self._measure(
                mbobs_list=mbobs_list,
                shear_bands=shear_bands,
                det_bands=det_bands,
                cat=cat,
                shear_str=shear_str,
                mfrac=kdata["mfrac"],
                bmask=kdata["bmask"],
                ormask=kdata["ormask"],
                psf_stats=kdata["psf_stats"],
            )

        return _result

    def _go_bands_with_color(self, shear_bands, mcal_res, det_bands):
        from . import detect

        _result = {}
        for shear_str, shear_mbobs in mcal_res.items():
            if not self._fitter_is_wavg[0]:
                raise RuntimeError(
                    "Color-dependent metadetect can only run if first"
                    " fitter is one of wmom, pgauss, or ksigma!"
                )

            # we first detect and get color of each detection
            cat, mbobs_list = self._do_detect(shear_mbobs, det_bands)
            nocolor_data = fit_mbobs_list_wavg(
                mbobs_list=mbobs_list,
                fitter=self._fitters[0],
                shear_bands=shear_bands,
                bmask_flags=self.get("bmask_flags", 0),
            )
            if nocolor_data is None:
                _result[shear_str] = None
                continue

            # now we map color to the mbobs for that color
            n = Namer(self._fitters[0].kind)
            col = n("band_flux")
            color_keys = [
                self.color_key_func(nocolor_data[col][i])
                for i in range(nocolor_data.shape[0])
            ]

            # now we remeasure the object at that mbobs
            color_data = []
            for i, color_key in enumerate(color_keys):
                kdata = self._get_mbobs_data(color_key, shear_bands)
                if kdata["mcal_res"] is None or kdata["mcal_res"][shear_str] is None:
                    continue

                _medsifier = detect.CatalogMEDSifier(
                    kdata["mcal_res"][shear_str],
                    cat['x'][i:i+1],
                    cat['y'][i:i+1],
                    cat['box_size'][i:i+1],
                )
                mbm = _medsifier.get_multiband_meds()
                mbobs_list = mbm.get_mbobs_list(
                    weight_type=self["meds"].get("weight_type", "weight"),
                )

                _data = self._measure(
                    mbobs_list=mbobs_list,
                    shear_bands=shear_bands,
                    det_bands=det_bands,
                    cat=cat[i:i+1],
                    shear_str=shear_str,
                    mfrac=kdata["mfrac"],
                    bmask=kdata["bmask"],
                    ormask=kdata["ormask"],
                    psf_stats=kdata["psf_stats"],
                )
                if _data is not None:
                    color_data.append(_data)

            if len(color_data) > 0:
                _result[shear_str] = np.hstack(color_data)
            else:
                _result[shear_str] = None

        return _result

    def _get_mbobs_data(self, key, shear_bands):
        logger.info("computing mbobs data: %s %s", key, shear_bands)

        if key is None:
            mbobs = self.mbobs
        else:
            mbobs = self.color_dep_mbobs[key]
            # the caching will miss cases where the input mbobs is passed as one
            # of the color dependent ones as well
            # thus we test if they are the same object and set the key to None if
            # this happens
            if mbobs is self.mbobs:
                key = None

        if not hasattr(self, "_mbobs_data_cache"):
            logger.info("set mbobs data caches")
            self._mbobs_data_cache = {}
            self._mcalpsf_data_cache = {}

        if key not in self._mbobs_data_cache:
            logger.info("key %s not in mbobs data cache", key)
            self._mbobs_data_cache[key] = {}
            self._mcalpsf_data_cache[key] = {}

            t0 = time.time()
            try:
                fitting.fit_all_psfs(mbobs, self.rng)
                _psf_fit_flags = 0
            except BootPSFFailure:
                _psf_fit_flags = procflags.PSF_FAILURE
            self._mcalpsf_data_cache[key]["psf_fit_flags"] = _psf_fit_flags
            logger.info("PSF fits took %s seconds", time.time() - t0)

            mcal_res = self._get_all_metacal(mbobs)
            self._mcalpsf_data_cache[key]["mcal_res"] = mcal_res

        sbkey = tuple(sorted(shear_bands))
        if sbkey not in self._mbobs_data_cache[key]:
            logger.info("shear_bands key %s not in mbobs data cache", sbkey)
            self._mbobs_data_cache[key][sbkey] = {}
            self._mbobs_data_cache[key][sbkey].update(
                self._mcalpsf_data_cache[key]
            )

            _mbobs = ngmix.MultiBandObsList()
            for band in shear_bands:
                _mbobs.append(mbobs[band])
            mfrac = self._get_mfrac(_mbobs)
            ormask, bmask = self._get_ormask_and_bmask(_mbobs)
            psf_stats = _get_psf_stats(
                _mbobs,
                self._mbobs_data_cache[key][sbkey]["psf_fit_flags"],
            )
            self._mbobs_data_cache[key][sbkey]["mfrac"] = mfrac
            self._mbobs_data_cache[key][sbkey]["bmask"] = bmask
            self._mbobs_data_cache[key][sbkey]["ormask"] = ormask
            self._mbobs_data_cache[key][sbkey]["psf_stats"] = psf_stats

        return self._mbobs_data_cache[key][sbkey]

    def _measure(
        self, *, mbobs_list, shear_bands, cat, shear_str, mfrac, bmask,
        ormask, psf_stats, det_bands,
    ):

        t0 = time.time()
        all_res = []
        for fitter, fwhm_reg, is_wavg, symm, coadd in zip(
            self._fitters, self._fwhm_regs,
            self._fitter_is_wavg, self._fitter_symmetrize,
            self._fitter_coadd,
        ):
            ft0 = time.time()
            if is_wavg:
                res = fit_mbobs_list_wavg(
                    mbobs_list=mbobs_list,
                    fitter=fitter,
                    shear_bands=shear_bands,
                    bmask_flags=self.get("bmask_flags", 0),
                    fwhm_reg=fwhm_reg,
                    symmetrize=symm,
                )
            else:
                res = fit_mbobs_list_joint(
                    mbobs_list=mbobs_list,
                    fitter_name=fitter,
                    shear_bands=shear_bands,
                    bmask_flags=self.get("bmask_flags", 0),
                    rng=self.rng,
                    symmetrize=symm,
                    coadd=coadd,
                )
            ft0 = time.time() - ft0
            logger.info(
                "fitter %s took %s seconds",
                fitter.kind
                if hasattr(fitter, "kind")
                else fitter,
                ft0,
            )
            all_res.append(res)

        res = combine_fit_res(all_res)

        if res is not None:
            res = self._add_positions_and_psf(
                cat=cat,
                res=res,
                shear_str=shear_str,
                mfrac=mfrac,
                bmask=bmask,
                ormask=ormask,
                psf_stats=psf_stats,
                det_bands=det_bands,
            )
        logger.info("src measurements took %s seconds", time.time() - t0)

        return res

    def _add_positions_and_psf(
        self, *, cat, res, shear_str, mfrac, bmask, ormask, psf_stats, det_bands,
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
            ('mfrac_img', 'f4'),
            ('ormask_noshear', 'i4'),
            ('mfrac_noshear', 'f4'),
            ('bmask_noshear', 'i4'),
            ("det_bands", "U%d" % MAX_NUM_SHEAR_BANDS),
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

        assert len(det_bands) <= MAX_NUM_SHEAR_BANDS
        newres["det_bands"] = "".join("%s" % b for b in sorted(det_bands))

        if 'psfrec_flags' not in res.dtype.names:
            newres['psfrec_flags'] = procflags.NO_ATTEMPT

        newres['psfrec_flags'][:] = psf_stats['flags']
        newres['psfrec_g'][:, 0] = psf_stats['g1']
        newres['psfrec_g'][:, 1] = psf_stats['g2']
        newres['psfrec_T'][:] = psf_stats['T']
        newres['mfrac_img'][:] = np.mean(mfrac)

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
                    fwhm=self.get("mfrac_fwhm", self._fwhms[0]),
                )

                newres["mfrac_noshear"] = measure_mfrac(
                    mfrac=mfrac,
                    x=newres["sx_col_noshear"],
                    y=newres["sx_row_noshear"],
                    box_sizes=cat["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", self._fwhms[0]),
                )
            else:
                newres["mfrac"] = 0
                newres["mfrac_noshear"] = 0

        return newres

    def _do_detect(self, mbobs, det_bands):
        """
        use a MEDSifier to run detection
        """
        from . import detect

        t0 = time.time()
        det_mbobs = ngmix.MultiBandObsList()
        for band in det_bands:
            det_mbobs.append(mbobs[band])

        medsifier = detect.MEDSifier(
            mbobs=det_mbobs,
            sx_config=self.get('sx', None),
            meds_config=self['meds'],
            nodet_flags=self['nodet_flags'],
        )

        if self._show:
            import descwl_coadd.vis
            descwl_coadd.vis.show_image(medsifier.seg)

        all_medsifier = detect.CatalogMEDSifier(
            mbobs,
            medsifier.cat['x'],
            medsifier.cat['y'],
            medsifier.cat['box_size'],
            seg=medsifier.seg,
            number=medsifier.cat['number'],
        )
        mbm = all_medsifier.get_multiband_meds()
        mbobs_list = mbm.get_mbobs_list(
            weight_type=self["meds"].get("weight_type", "weight"),
        )
        logger.info("detect took %s seconds", time.time() - t0)

        return medsifier.cat, mbobs_list

    def _get_all_metacal(self, mbobs):
        """
        get the sheared versions of the observations
        """
        t0 = time.time()
        try:
            odict = ngmix.metacal.get_all_metacal(
                mbobs,
                rng=self.rng,
                **self['metacal']
            )
        except BootPSFFailure:
            odict = None
        logger.info("metacal took %s seconds", time.time() - t0)

        if self._show and odict is not None:
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
        flags = procflags.PSF_FAILURE | global_flags
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
