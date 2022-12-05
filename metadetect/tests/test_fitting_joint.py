import numpy as np
import ngmix

import pytest

from .sim import make_mbobs_sim
from ..fitting import (
    fit_mbobs_admom,
    fit_mbobs_list_joint,
    make_coadd_obs,
    get_admom_runner,
    symmetrize_obs_weights,
    fit_mbobs_gauss,
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
    ran_one = False

    if case == "shear_bands_bad_oneband":
        mbobs = make_mbobs_sim(45, 1, wcs_var_scale=0)
        coadd_obs, flags = make_coadd_obs(
            mbobs, shear_bands=[10]
        )
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        ran_one = True
    elif case == "shear_bands_bad":
        mbobs = make_mbobs_sim(45, 3, wcs_var_scale=0)
        coadd_obs, flags = make_coadd_obs(
            mbobs, shear_bands=[10]
        )
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        ran_one = True
    elif case == "shear_bands_outofrange":
        mbobs = make_mbobs_sim(45, 3, wcs_var_scale=0)
        coadd_obs, flags = make_coadd_obs(
            mbobs, shear_bands=[10, 13, 0]
        )
        assert coadd_obs is None
        assert flags == procflags.INCONSISTENT_BANDS
        ran_one = True
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
        ran_one = True
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
        ran_one = True
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
        ran_one = True
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
        ran_one = True
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
        ran_one = True
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
        ran_one = True
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
            ran_one = True
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
        ran_one = True
    else:
        assert False, f"case {case} not found!"

    assert ran_one, "No tests ran!"


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


@pytest.mark.parametrize("coadd", [True, False])
@pytest.mark.parametrize("symmetrize", [True, False])
@pytest.mark.parametrize("shear_bands", [None, [0, 1], [2, 3, 1]])
@pytest.mark.parametrize("fname", ["am", "admom", "gauss"])
def test_fit_mbobs_list_joint_seeding(shear_bands, fname, coadd, symmetrize):
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
        coadd=coadd,
        symmetrize=symmetrize,
    )

    rng1 = np.random.RandomState(seed=4235)
    res1 = fit_mbobs_list_joint(
        mbobs_list=mbobs_list,
        fitter_name=fname,
        bmask_flags=0,
        rng=rng1,
        shear_bands=shear_bands,
        coadd=coadd,
        symmetrize=symmetrize,
    )
    for col in res.dtype.names:
        np.testing.assert_array_equal(
            res[col],
            res1[col],
            err_msg=col,
        )


@pytest.mark.parametrize("case", [
    "missing_band",
    "too_many_bands",
    "zero_weights",
    "edge_hit",
    "coadd_flags",
])
def test_fit_mbobs_admom_input_errors(case):
    rng = np.random.RandomState(seed=211324)
    ran_one = False

    if case == "missing_band":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1] = ngmix.ObsList()
        res = fit_mbobs_admom(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["am_flags"] == (procflags.NO_ATTEMPT | procflags.MISSING_BAND), (
            procflags.get_procflags_str(res["am_flags"][0])
        )
    elif case == "too_many_bands":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        ol = ngmix.ObsList()
        ol.append(mbobs[1][0])
        ol.append(mbobs[1][0])
        mbobs[1] = ol
        res = fit_mbobs_admom(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["am_flags"] == (
            procflags.NO_ATTEMPT | procflags.INCONSISTENT_BANDS
        ), (
            procflags.get_procflags_str(res["am_flags"][0])
        )
    elif case == "zero_weights":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=np.zeros_like(mbobs[1][0].image),
            ignore_zero_weight=False,
            bmask=mbobs[1][0].bmask,
        )
        res = fit_mbobs_admom(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["am_flags"] == (procflags.NO_ATTEMPT | procflags.ZERO_WEIGHTS), (
            procflags.get_procflags_str(res["am_flags"][0])
        )
    elif case == "edge_hit":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=mbobs[1][0].weight,
            bmask=mbobs[1][0].bmask + 10,
        )
        res = fit_mbobs_admom(
            mbobs=mbobs,
            bmask_flags=10,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["am_flags"] == (procflags.NO_ATTEMPT | procflags.EDGE_HIT), (
            procflags.get_procflags_str(res["am_flags"][0])
        )
    elif case == "coadd_flags":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=mbobs[1][0].weight,
            bmask=mbobs[1][0].bmask,
            ormask=mbobs[1][0].ormask,
            noise=mbobs[1][0].noise,
            # missing mfrac so coadd fails
        )
        res = fit_mbobs_admom(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["am_flags"] == (
            procflags.NO_ATTEMPT | procflags.INCONSISTENT_BANDS
        ), (
            procflags.get_procflags_str(res["am_flags"][0])
        )
    else:
        assert False, f"case {case} not found!"

    assert ran_one, "No tests ran!"


@pytest.mark.parametrize("shear_bands", [[0], [2]])
def test_fit_mbobs_admom_oneband(shear_bands):
    rng = np.random.RandomState(seed=211324)
    mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
    res = fit_mbobs_admom(
        mbobs=mbobs,
        bmask_flags=0,
        rng=rng,
        shear_bands=shear_bands,
    )

    # make sure things are seeded and everything is copied out
    # correctly
    assert res["am_flags"] == (
        res["am_psf_flags"] | res["am_obj_flags"]
    )

    rng = np.random.RandomState(seed=211324)
    fitter = get_admom_runner(rng)
    pres = fitter.go(mbobs[shear_bands[0]][0].psf)
    np.testing.assert_allclose(
        res["am_psf_T"][0], pres["T"]
    )
    np.testing.assert_allclose(
        res["am_psf_g"][0], pres["e"]
    )

    sobs = symmetrize_obs_weights(mbobs[shear_bands[0]][0])
    gres = fitter.go(sobs)
    np.testing.assert_array_equal(res["am_T_flags"][0], gres["T_flags"])
    np.testing.assert_array_equal(res["am_T"][0], gres["T"])
    np.testing.assert_array_equal(res["am_T_err"][0], gres["T_err"])
    np.testing.assert_array_equal(
        res["am_T_ratio"][0],
        gres["T"]/pres["T"],
    )

    np.testing.assert_array_equal(res["am_obj_flags"][0], gres["flags"])
    np.testing.assert_array_equal(res["am_s2n"][0], gres["s2n"])
    np.testing.assert_array_equal(res["am_g"][0], gres["e"])
    np.testing.assert_array_equal(res["am_g_cov"][0], gres["e_cov"])


@pytest.mark.parametrize("shear_bands", [[0], [2], None, [1, 3, 2]])
def test_fit_mbobs_admom_smoke(shear_bands):
    rng = np.random.RandomState(seed=211324)
    mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
    res = fit_mbobs_admom(
        mbobs=mbobs,
        bmask_flags=0,
        rng=rng,
        shear_bands=shear_bands,
    )

    # none of these fits should fail
    if shear_bands is None:
        shear_bands = list(range(len(mbobs)))

    for col in res.dtype.names:
        if col.endswith("band_flux_flags"):
            assert np.all(res[col] == procflags.NO_ATTEMPT), (col, res[col])
        elif "band_flux" in col:
            assert np.all(np.isnan(res[col])), (col, res[col])
        elif col == "shear_bands":
            assert np.all(
                res[col]
                == "".join(f"{sb}" for sb in sorted(shear_bands))
            ), (col, res[col])
        elif not col.endswith("flags"):
            assert np.all(np.isfinite(res[col])), (col, res[col])
        else:
            assert np.all(res[col] == 0), (col, res[col])


@pytest.mark.parametrize("shear_bands", [[0], [2], None, [1, 3, 2]])
def test_fit_mbobs_gauss_smoke(shear_bands):
    rng = np.random.RandomState(seed=211324)
    mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
    res = fit_mbobs_gauss(
        mbobs=mbobs,
        bmask_flags=0,
        rng=rng,
        shear_bands=shear_bands,
    )

    # none of these fits should fail
    if shear_bands is None:
        shear_bands = list(range(len(mbobs)))

    for col in res.dtype.names:
        if col.endswith("band_flux_flags"):
            assert np.all(res[col] == procflags.NO_ATTEMPT), (col, res[col])
        elif "band_flux" in col:
            assert np.all(np.isnan(res[col])), (col, res[col])
        elif col == "shear_bands":
            assert np.all(
                res[col]
                == "".join(f"{sb}" for sb in sorted(shear_bands))
            ), (col, res[col])
        elif not col.endswith("flags"):
            assert np.all(np.isfinite(res[col])), (col, res[col])
        else:
            assert np.all(res[col] == 0), (col, res[col])


@pytest.mark.parametrize("coadd", [True, False])
@pytest.mark.parametrize("case", [
    "missing_band",
    "too_many_bands",
    "zero_weights",
    "edge_hit",
    "shear_bands",
    "coadd_flags",
])
def test_fit_mbobs_gauss_input_errors(case, coadd):
    rng = np.random.RandomState(seed=211324)
    ran_one = False

    if case == "missing_band":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1] = ngmix.ObsList()
        res = fit_mbobs_gauss(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["gauss_flags"] == (procflags.NO_ATTEMPT | procflags.MISSING_BAND), (
            procflags.get_procflags_str(res["gauss_flags"][0])
        )
    elif case == "too_many_bands":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        ol = ngmix.ObsList()
        ol.append(mbobs[1][0])
        ol.append(mbobs[1][0])
        mbobs[1] = ol
        res = fit_mbobs_gauss(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["gauss_flags"] == (
            procflags.NO_ATTEMPT | procflags.INCONSISTENT_BANDS
        ), (
            procflags.get_procflags_str(res["gauss_flags"][0])
        )
    elif case == "zero_weights":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=np.zeros_like(mbobs[1][0].image),
            ignore_zero_weight=False,
            bmask=mbobs[1][0].bmask,
        )
        res = fit_mbobs_gauss(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["gauss_flags"] == (procflags.NO_ATTEMPT | procflags.ZERO_WEIGHTS), (
            procflags.get_procflags_str(res["gauss_flags"][0])
        )
    elif case == "edge_hit":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=mbobs[1][0].weight,
            bmask=mbobs[1][0].bmask + 10,
        )
        res = fit_mbobs_gauss(
            mbobs=mbobs,
            bmask_flags=10,
            rng=rng,
            shear_bands=None,
        )
        ran_one = True
        assert res["gauss_flags"] == (procflags.NO_ATTEMPT | procflags.EDGE_HIT), (
            procflags.get_procflags_str(res["gauss_flags"][0])
        )
    elif case == "shear_bands":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        res = fit_mbobs_gauss(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=[1, 2, 10],
        )
        ran_one = True
        assert res["gauss_flags"] == (
            procflags.NO_ATTEMPT | procflags.INCONSISTENT_BANDS
        ), (
            procflags.get_procflags_str(res["gauss_flags"][0])
        )
    elif case == "coadd_flags":
        mbobs = make_mbobs_sim(45, 4, wcs_var_scale=0)
        mbobs[1][0] = ngmix.Observation(
            image=mbobs[1][0].image,
            jacobian=mbobs[1][0].jacobian,
            psf=mbobs[1][0].psf,
            weight=mbobs[1][0].weight,
            bmask=mbobs[1][0].bmask,
            ormask=mbobs[1][0].ormask,
            noise=mbobs[1][0].noise,
            # missing mfrac so coadd fails
        )
        res = fit_mbobs_gauss(
            mbobs=mbobs,
            bmask_flags=0,
            rng=rng,
            shear_bands=None,
            coadd=True,
        )
        ran_one = True
        if coadd:
            assert res["gauss_flags"] == (
                procflags.NO_ATTEMPT | procflags.INCONSISTENT_BANDS
            ), (
                procflags.get_procflags_str(res["gauss_flags"][0])
            )
        else:
            assert res["gauss_flags"] == 0, (
                procflags.get_procflags_str(res["gauss_flags"][0])
            )
    else:
        assert False, f"case {case} not found!"

    assert ran_one, "No tests ran!"
