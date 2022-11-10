import numpy as np
import ngmix

import pytest

from .sim import make_mbobs_sim
from ..fitting import (
    # fit_mbobs_admom,
    # fit_mbobs_list_joint,
    make_coadd_obs,
)
from .. import procflags


@pytest.mark.parametrize("case", [
    "shear_bands_bad_oneband",
    "shear_bands_bad",
    "shear_bands_outofrange",
    "bad_img_jacob",
    "bad_img_shape",
    "missing_psf",
    "bad_psf_jacob",
    "bad_psf_img_shape",
    "zero_weights",
    "missing_attrs",
    "disjoint_weights",
])
def test_make_coadd_obs_errors(case):
    if case == "shear_bands_bad_oneband":
        mbobs = make_mbobs_sim(45, 1, wcs_var_scale=0)
        coadd_obs, flags = make_coadd_obs(
            mbobs, shear_bands=[10]
        )
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
    elif case == "shear_bands_bad":
        mbobs = make_mbobs_sim(45, 3, wcs_var_scale=0)
        coadd_obs, flags = make_coadd_obs(
            mbobs, shear_bands=[10]
        )
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
    elif case == "shear_bands_outofrange":
        mbobs = make_mbobs_sim(45, 3, wcs_var_scale=0)
        coadd_obs, flags = make_coadd_obs(
            mbobs, shear_bands=[10, 13, 0]
        )
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
    elif case == "bad_img_jacob":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[0][0].image,
            jacobian=ngmix.DiagonalJacobian(
                scale=0.25,
                row=10,
                col=9,
            ),
            psf=mbobs[1][0].psf,
        )
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
        assert coadd_obs is not None
        assert flags == 0
    elif case == "bad_img_shape":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=np.zeros((5, 5)),
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
        )
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
        assert coadd_obs is not None
        assert flags == 0
    elif case == "missing_psf":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
        )
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
        assert coadd_obs is not None
        assert flags == 0
    elif case == "bad_psf_jacob":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0].psf = ngmix.Observation(
            image=mbobs[1][0].psf.image,
            jacobian=ngmix.DiagonalJacobian(
                scale=0.25,
                row=10,
                col=9,
            ),
        )
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
        assert coadd_obs is not None
        assert flags == 0
    elif case == "bad_psf_img_shape":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0].psf = ngmix.Observation(
            image=np.zeros((5, 5)),
            jacobian=mbobs[1][0].psf.jacobian,
        )
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
        assert coadd_obs is not None
        assert flags == 0
    elif case == "zero_weights":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=np.zeros_like(mbobs[1][0].image),
            ignore_zero_weight=False,
        )
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
        assert coadd_obs is None
        assert flags == procflags.ZERO_WEIGHTS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
        assert coadd_obs is None
        assert flags == procflags.ZERO_WEIGHTS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
        assert coadd_obs is not None
        assert flags == 0
    elif case == "missing_attrs":
        for attr in ["mfrac", "noise", "bmask", "ormask"]:
            mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
            kwargs = {
                "mfrac": mbobs[1][0].mfrac,
                "noise": mbobs[1][0].noise,
                "bmask": mbobs[1][0].bmask,
                "ormask": mbobs[1][0].ormask,
            }
            kwargs.pop(attr)
            mbobs[1][0] = ngmix.Observation(
                image=mbobs[1][0].image,
                jacobian=mbobs[1][0].jacobian,
                psf=mbobs[1][0].psf,
                weight=mbobs[1][0].weight,
                **kwargs,
            )
            coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
            assert coadd_obs is None
            assert flags == procflags.INCONSISTENT_BANDS
            coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
            assert coadd_obs is None
            assert flags == procflags.INCONSISTENT_BANDS
            coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
            assert coadd_obs is not None
            assert flags == 0
    elif case == "disjoint_weights":
        # if two images have disjoint weight maps where no non-zero
        # areas overlap, then we cannot coadd
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        wmsk = np.zeros_like(mbobs[1][0].image)
        wmsk[:10, :] = 1
        mbobs[0][0] = ngmix.Observation(
            image=mbobs[0][0].image,
            jacobian=mbobs[0][0].jacobian,
            psf=mbobs[0][0].psf,
            weight=mbobs[0][0].weight * (1.0 - wmsk),
            mfrac=mbobs[0][0].mfrac,
            ormask=mbobs[0][0].ormask,
            bmask=mbobs[0][0].bmask,
            noise=mbobs[0][0].noise,
        )
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=mbobs[1][0].weight * wmsk,
            mfrac=mbobs[1][0].mfrac,
            ormask=mbobs[1][0].ormask,
            bmask=mbobs[1][0].bmask,
            noise=mbobs[1][0].noise,
        )
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
        assert coadd_obs is None
        assert flags == procflags.ZERO_WEIGHTS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
        assert coadd_obs is None
        assert flags == procflags.ZERO_WEIGHTS
        coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[0, 2, 3])
        assert coadd_obs is not None
        assert flags == 0
    else:
        assert False, f"case {case} not found!"


# def fit_mbobs_list_joint(
#     *, mbobs_list, fitter_name, bmask_flags, rng, shear_bands=None,
# ):
#     """Fit the ojects in a list of ngmix.MultiBandObsList using a joint fitter.

#     The fitter is run per object on the bands used for shear. A follow-up method
#     is called to produce per-band fluxes if the fitter supports it.

#     Parameters
#     ----------
#     mbobs_list : a list of ngmix.MultiBandObsList
#         The observations to use for shear measurement.
#     fitter_name : str
#         The name of the fitter to use.
#     bmask_flags : int
#         Observations with these bits set in the bmask are not fit.
#     shear_bands : list of int, optional
#         A list of indices into each mbobs that denotes which band is used for shear.
#         Default is to use all bands.
#     rng : np.random.RandomState
#         Random state for fitting.

#     Returns
#     -------
#     res : np.ndarray
#         A structured array of the fitting results.
#     """
#     if fitter_name in ["am", "admom"]:
#         fit_func = fit_mbobs_admom
#     else:
#         raise RuntimeError("Joint fitter '%s' not recognized!" % fitter_name)

#     res = []
#     for i, mbobs in enumerate(mbobs_list):
#         _res = fit_func(
#             mbobs=mbobs,
#             bmask_flags=bmask_flags,
#             shear_bands=shear_bands,
#             rng=rng,
#         )
#         res.append(_res)

#     if len(res) > 0:
#         return np.hstack(res)
#     else:
#         return None


# def fit_mbobs_admom(
#     *,
#     mbobs,
#     bmask_flags,
#     rng,
#     shear_bands=None,
# ):
#     """Fit a multiband obs using adaptive moments.

#     This function forms a coadd of the shear bands and then runs
#     adaptive moments on the coadd.

#     Parameters
#     ----------
#     mbobs : ngmix.MultiBandObsList
#         The observation to use for shear measurement.
#     bmask_flags : int
#         Observations with these bits set in the bmask are not fit.
#     rng : np.random.RandomState
#         Random state for fitting.
#     shear_bands : list of int, optional
#         A list of indices into each mbobs that denotes which band is used for shear.
#         Default is to use all bands.

#     Returns
#     -------
#     res : np.ndarray
#         A structured array of the fitting results.
#     """
#     fitter = get_admom_fitter(rng)
#     nband = len(mbobs)
#     res = get_wavg_output_struct(nband, "am", shear_bands=shear_bands)

#     flags = 0
#     for obslist in mbobs:
#         if len(obslist) == 0:
#             flags |= procflags.MISSING_BAND
#             continue

#         if len(obslist) > 1:
#             flags |= procflags.INCONSISTENT_BANDS
#             continue

#         for obs in obslist:
#             if not np.any(obs.weight > 0):
#                 flags |= procflags.ZERO_WEIGHTS

#             if np.any((obs.bmask & bmask_flags) != 0):
#                 flags |= procflags.EDGE_HIT

#     if flags == 0:
#         # first we coadd the shear bands
#         coadd_obs, coadd_flags = make_coadd_obs(mbobs, shear_bands=shear_bands)
#         flags |= coadd_flags

#     if flags == 0:
#         # then fit the PSF
#         pres = fitter.go(coadd_obs.psf)
#         res["am_psf_flags"] = pres["flags"]
#         if pres["flags"] == 0:
#             res["am_psf_g"] = pres["e"]
#             res["am_psf_T"] = pres["T"]

#         # then fit the object
#         sym_coadd_obs = symmetrize_obs_weights(coadd_obs)
#         gres = fitter.go(sym_coadd_obs)
#         res["am_T_flags"] = gres["T_flags"]
#         if gres["T_flags"] == 0:
#             res["am_T"] = gres["T"]
#             res["am_T_err"] = gres["T_err"]
#             if pres["flags"] == 0:
#                 res["am_T_ratio"] = res["am_T"] / res["am_psf_T"]

#         res["am_obj_flags"] = gres["flags"]
#         if gres["flags"] == 0:
#             res["am_s2n"] = gres["s2n"]
#             res["am_g"] = gres["e"]
#             res["am_g_cov"] = gres["e_cov"]

#         # this replaces the flags so they are zero and unsets the default of
#         # no attempt
#         res["am_flags"] = (res["am_psf_flags"] | res["am_obj_flags"])
#     else:
#         # this branch ensures noattempt remains set
#         res["am_flags"] |= flags

#     return res
