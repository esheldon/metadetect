import numpy as np
import ngmix

import pytest

from .sim import make_mbobs_sim
from ..fitting import (
    fit_mbobs_admom,
    fit_mbobs_list_joint,
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


def test_make_coadd_obs_single():
    mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)

    coadd_obs, flags = make_coadd_obs(mbobs[:1])
    assert coadd_obs is mbobs[0][0]
    assert flags == 0

    coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=[2])
    assert coadd_obs is mbobs[2][0]
    assert flags == 0


def test_make_coadd_obs_symmetric():
    mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)

    coadd_obs01, flags = make_coadd_obs(mbobs, shear_bands=[0, 1])
    assert flags == 0

    coadd_obs10, flags = make_coadd_obs(mbobs, shear_bands=[1, 0])
    assert flags == 0

    assert repr(coadd_obs01.jacobian) == repr(coadd_obs10.jacobian)
    assert coadd_obs01.meta == mbobs[1][0].meta
    assert coadd_obs10.meta == mbobs[0][0].meta
    for attr in [
        "image", "weight",
        "bmask", "ormask", "noise", "mfrac",
    ]:
        assert np.array_equal(
            getattr(coadd_obs01, attr),
            getattr(coadd_obs10, attr),
        )

    assert repr(coadd_obs01.psf.jacobian) == repr(coadd_obs10.psf.jacobian)
    assert coadd_obs01.psf.meta == mbobs[1][0].psf.meta
    assert coadd_obs10.psf.meta == mbobs[0][0].psf.meta
    for attr in ["image", "weight"]:
        assert np.array_equal(
            getattr(coadd_obs01.psf, attr),
            getattr(coadd_obs10.psf, attr),
        )


def _or_arrays(arrs):
    res = arrs[0].copy()
    for arr in arrs[1:]:
        res |= arr

    return res


@pytest.mark.parametrize("shear_bands", [None, [0, 1], [3, 1, 2]])
def test_make_coadd_obs_shear_bands(shear_bands):
    mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)

    coadd_obs, flags = make_coadd_obs(mbobs, shear_bands=shear_bands)
    assert flags == 0

    # make sure different shear bands give different answers
    coadd_obs_none, flags = make_coadd_obs(mbobs, shear_bands=None)
    assert flags == 0

    if shear_bands is not None:
        assert not np.array_equal(
            coadd_obs.image,
            coadd_obs_none.image
        )
    else:
        assert np.array_equal(
            coadd_obs.image,
            coadd_obs_none.image
        )

    if shear_bands is None:
        shear_bands = list(range(len(mbobs)))

    # the coadded image should have higher s/n
    assert all(
        coadd_obs.get_s2n()
        > mbobs[sb][0].get_s2n() for sb in shear_bands
    )

    # total flux should be close
    fluxes = list(
        mbobs[sb][0].image.sum()
        for sb in shear_bands
    )
    assert np.allclose(
        coadd_obs.image.sum(),
        fluxes,
        atol=0,
        rtol=0.2,
    ), (coadd_obs.image.sum(), fluxes)

    # psf should sum to unity
    assert np.allclose(coadd_obs.psf.image.sum(), 1)

    # jacobians and image shapes should be the same
    assert repr(coadd_obs.jacobian) == repr(mbobs[0][0].jacobian)
    assert coadd_obs.image.shape == mbobs[0][0].image.shape
    assert repr(coadd_obs.psf.jacobian) == repr(mbobs[0][0].psf.jacobian)
    assert coadd_obs.psf.image.shape == mbobs[0][0].psf.image.shape
    assert coadd_obs.meta == mbobs[shear_bands[-1]][0].meta

    assert np.array_equal(
        coadd_obs.bmask,
        _or_arrays([mbobs[sb][0].bmask for sb in shear_bands]),
    )
    assert np.array_equal(
        coadd_obs.ormask,
        _or_arrays([mbobs[sb][0].ormask for sb in shear_bands]),
    )

    wgts = [np.median(mbobs[sb][0].weight) for sb in shear_bands]
    wgts = np.array(wgts)
    wgts /= np.sum(wgts)
    # weights should be different
    assert not np.allclose(wgts[0], wgts)

    # check the actual values in coadded attributes
    image = np.zeros_like(mbobs[0][0].image)
    weight = np.zeros_like(mbobs[0][0].image)
    mfrac = np.zeros_like(mbobs[0][0].image)
    noise = np.zeros_like(mbobs[0][0].image)
    psf_image = np.zeros_like(mbobs[0][0].psf.image)
    for i, sb in enumerate(shear_bands):
        image += mbobs[sb][0].image * wgts[i]
        mfrac += mbobs[sb][0].mfrac * wgts[i]
        noise += mbobs[sb][0].noise * wgts[i]
        psf_image += mbobs[sb][0].psf.image * wgts[i]
        weight += wgts[i]**2 / mbobs[sb][0].weight
    weight = 1.0 / weight
    assert np.allclose(image, coadd_obs.image)
    assert np.allclose(weight, coadd_obs.weight)
    assert np.allclose(mfrac, coadd_obs.mfrac)
    assert np.allclose(noise, coadd_obs.noise)
    assert np.allclose(psf_image, coadd_obs.psf.image)


def test_fit_mbobs_list_joint_errors():
    mbobs_list = [
        make_mbobs_sim(45, 3, wcs_var_scale=0),
        make_mbobs_sim(46, 3, wcs_var_scale=0),
    ]
    rng = np.random.RandomState(seed=4235)
    with pytest.raises(RuntimeError) as err:
        fit_mbobs_list_joint(
            mbobs_list=mbobs_list,
            fitter_name="blah",
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
    assert "fitter 'blah'" in str(err.value)


def test_fit_mbobs_list_joint_empty():
    mbobs_list = []
    rng = np.random.RandomState(seed=4235)
    res = fit_mbobs_list_joint(
        mbobs_list=mbobs_list,
        fitter_name="am",
        bmask_flags=0,
        rng=rng,
        shear_bands=None,
    )
    assert res is None


@pytest.mark.parametrize("shear_bands", [None, [0, 1], [2, 3, 1]])
@pytest.mark.parametrize("fname", ["am", "admom"])
def test_fit_mbobs_list_joint_fits_all(shear_bands, fname):
    mbobs_list = [
        make_mbobs_sim(45, 4, wcs_var_scale=0),
        make_mbobs_sim(46, 4, wcs_var_scale=0),
        make_mbobs_sim(47, 4, wcs_var_scale=0),
    ]
    rng = np.random.RandomState(seed=4235)
    res = fit_mbobs_list_joint(
        mbobs_list=mbobs_list,
        fitter_name=fname,
        bmask_flags=0,
        rng=rng,
        shear_bands=shear_bands,
    )
    assert res.shape == (3,)

    if shear_bands is None:
        shear_bands = list(range(4))
    assert np.all(
        res["shear_bands"]
        == "".join(str(sb) for sb in sorted(shear_bands))
    )

    rng = np.random.RandomState(seed=4235)
    for i in range(3):
        res1 = fit_mbobs_admom(
            mbobs=mbobs_list[i],
            bmask_flags=0,
            rng=rng,
            shear_bands=shear_bands,
        )

        for col in res.dtype.names:
            np.testing.assert_array_equal(res[i:i+1][col], res1[col])


@pytest.mark.parametrize("shear_bands", [None, [0, 1], [2, 3, 1]])
@pytest.mark.parametrize("fname", ["am", "admom"])
def test_fit_mbobs_list_joint_seeding(shear_bands, fname):
    mbobs_list = [
        make_mbobs_sim(45, 4, wcs_var_scale=0),
        make_mbobs_sim(46, 4, wcs_var_scale=0),
        make_mbobs_sim(47, 4, wcs_var_scale=0),
    ]
    rng = np.random.RandomState(seed=4235)
    res = fit_mbobs_list_joint(
        mbobs_list=mbobs_list,
        fitter_name=fname,
        bmask_flags=0,
        rng=rng,
        shear_bands=shear_bands,
    )

    rng1 = np.random.RandomState(seed=4235)
    res1 = fit_mbobs_list_joint(
        mbobs_list=mbobs_list,
        fitter_name=fname,
        bmask_flags=0,
        rng=rng1,
        shear_bands=shear_bands,
    )
    for col in res.dtype.names:
        np.testing.assert_array_equal(res[col], res1[col])


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
