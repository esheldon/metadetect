import numpy as np

import pytest

from ngmix.gaussmom import GaussMom
import ngmix

from .sim import make_mbobs_sim
from ..fitting import fit_mbobs_wavg, _combine_fit_results_wavg
from .. import procflags


def _print_res(res):
    print("", flush=True)
    for name in res.dtype.names:
        print("    %s:" % name, res[name], flush=True)


def test_fitting_fit_mbobs_wavg_flagging_nodata():
    mbobs = make_mbobs_sim(45, 4)
    mbobs[1] = ngmix.ObsList()
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        nonshear_mbobs=None,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    mbobs[1] = ngmix.ObsList()
    nonshear_mbobs = make_mbobs_sim(45, 3)
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    nonshear_mbobs = make_mbobs_sim(45, 3)
    nonshear_mbobs[1] = ngmix.ObsList()
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 5]))
        for i in [0, 1, 2, 3, 4, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))


def test_fitting_fit_mbobs_wavg_flagging_edge():
    bmask_flags = 2**8
    other_flags = 2**3
    mbobs = make_mbobs_sim(45, 4)
    with mbobs[1][0].writeable():
        mbobs[1][0].bmask[2, 3] = bmask_flags
        mbobs[1][0].bmask[3, 1] = other_flags
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        nonshear_mbobs=None,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    with mbobs[1][0].writeable():
        mbobs[1][0].bmask[2, 3] = bmask_flags
        mbobs[1][0].bmask[3, 1] = other_flags
    nonshear_mbobs = make_mbobs_sim(45, 3)
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    nonshear_mbobs = make_mbobs_sim(45, 3)
    with nonshear_mbobs[1][0].writeable():
        nonshear_mbobs[1][0].bmask[2, 3] = bmask_flags
        nonshear_mbobs[1][0].bmask[3, 1] = other_flags
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 5]))
        for i in [0, 1, 2, 3, 4, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))


def test_fitting_fit_mbobs_wavg_flagging_zeroweight():
    mbobs = make_mbobs_sim(45, 4)
    with mbobs[1][0].writeable():
        mbobs[1][0].ignore_zero_weight = False
        mbobs[1][0].weight[:, :] = 0
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        nonshear_mbobs=None,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    with mbobs[1][0].writeable():
        mbobs[1][0].ignore_zero_weight = False
        mbobs[1][0].weight[:, :] = 0
    nonshear_mbobs = make_mbobs_sim(45, 3)
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    nonshear_mbobs = make_mbobs_sim(45, 3)
    with nonshear_mbobs[1][0].writeable():
        nonshear_mbobs[1][0].ignore_zero_weight = False
        nonshear_mbobs[1][0].weight[:, :] = 0
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 5]))
        for i in [0, 1, 2, 3, 4, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))


def test_fitting_fit_mbobs_wavg_flagging_combined():
    mbobs = make_mbobs_sim(45, 4)
    bmask_flags = 2**8
    other_flags = 2**3
    mbobs[0] = ngmix.ObsList()
    with mbobs[1][0].writeable():
        mbobs[1][0].bmask[2, 3] = bmask_flags
        mbobs[1][0].bmask[3, 1] = other_flags
    with mbobs[2][0].writeable():
        mbobs[2][0].ignore_zero_weight = False
        mbobs[2][0].weight[:, :] = 0
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        nonshear_mbobs=None,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, 3]))
        for i in [0, 1, 2]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    bmask_flags = 2**8
    other_flags = 2**3
    mbobs[0] = ngmix.ObsList()
    with mbobs[1][0].writeable():
        mbobs[1][0].bmask[2, 3] = bmask_flags
        mbobs[1][0].bmask[3, 1] = other_flags
    with mbobs[2][0].writeable():
        mbobs[2][0].ignore_zero_weight = False
        mbobs[2][0].weight[:, :] = 0
    nonshear_mbobs = make_mbobs_sim(45, 3)
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        for i in [3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
        for i in [0, 1, 2]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))

    mbobs = make_mbobs_sim(45, 4)
    nonshear_mbobs = make_mbobs_sim(45, 3)
    bmask_flags = 2**8
    other_flags = 2**3
    nonshear_mbobs[0] = ngmix.ObsList()
    with nonshear_mbobs[1][0].writeable():
        nonshear_mbobs[1][0].ignore_zero_weight = False
        nonshear_mbobs[1][0].weight[:, :] = 0
    with nonshear_mbobs[2][0].writeable():
        nonshear_mbobs[2][0].bmask[2, 3] = bmask_flags
        nonshear_mbobs[2][0].bmask[3, 1] = other_flags
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        nonshear_mbobs=nonshear_mbobs,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.MISSING_BAND) != 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        for i in [0, 1, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
        for i in [4, 5, 6]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))


@pytest.mark.parametrize("purpose,kwargs,psf_flags,model_flags,flux_flags", [
    (
        "no data at all",
        dict(
            all_res=[],
            all_psf_res=[],
            all_is_shear_band=[],
            all_wgts=[],
            all_flags=[],
        ),
        ["MISSING_BAND"],
        ["MISSING_BAND", "PSF_FAILURE"],
        ["MISSING_BAND"],
    ),
    (
        "everything failed",
        dict(
            all_res=[None, None, None, None],
            all_psf_res=[None, None, None, None],
            all_is_shear_band=[True, True, False, False],
            all_wgts=[0, 0, 0, 0],
            all_flags=[0, 0, 0, 0],
        ),
        ["MISSING_BAND", "ZERO_WEIGHTS"],
        ["MISSING_BAND", "PSF_FAILURE", "ZERO_WEIGHTS"],
        ["MISSING_BAND"],
    ),

    (
        "everything failed w/ input flags that should be in the output",
        dict(
            all_res=[None, None, None, None],
            all_psf_res=[None, None, None, None],
            all_is_shear_band=[True, True, False, False],
            all_wgts=[0, 0, 0, 0],
            all_flags=[1, 0, 1, 0],
        ),
        ["MISSING_BAND", "ZERO_WEIGHTS", "NO_ATTEMPT"],
        ["MISSING_BAND", "PSF_FAILURE", "NO_ATTEMPT", "ZERO_WEIGHTS"],
        ["MISSING_BAND", "NO_ATTEMPT"],
    ),
    (
        "we mark weights zero vs not for failures",
        dict(
            all_res=[None, None, None, None],
            all_psf_res=[None, None, None, None],
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["MISSING_BAND"],
        ["MISSING_BAND", "PSF_FAILURE"],
        ["MISSING_BAND"],
    ),
    (
        "everything is fine one band",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 1,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 1,
            all_is_shear_band=[True],
            all_wgts=[1],
            all_flags=[0],
        ),
        [],
        [],
        [],
    ),
    (
        "everything is fine for more than one band",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [],
    ),
    (
        "extra shear bands",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 2,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 1,
            all_is_shear_band=[True],
            all_wgts=[1],
            all_flags=[0],
        ),
        ["INCONSISTENT_BANDS"],
        ["INCONSISTENT_BANDS", "PSF_FAILURE"],
        ["INCONSISTENT_BANDS"],
    ),
    (
        "extra PSF bands",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 2,
            all_is_shear_band=[True],
            all_wgts=[1],
            all_flags=[0],
        ),
        ["INCONSISTENT_BANDS"],
        ["INCONSISTENT_BANDS", "PSF_FAILURE"],
        ["INCONSISTENT_BANDS"],
    ),
    (
        "extra weights",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}],
            all_is_shear_band=[True],
            all_wgts=[1, 1],
            all_flags=[0],
        ),
        ["INCONSISTENT_BANDS"],
        ["INCONSISTENT_BANDS", "PSF_FAILURE"],
        ["INCONSISTENT_BANDS"],
    ),
    (
        "extra flags",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}],
            all_is_shear_band=[True],
            all_wgts=[1],
            all_flags=[0, 0],
        ),
        ["INCONSISTENT_BANDS"],
        ["INCONSISTENT_BANDS", "PSF_FAILURE"],
        ["INCONSISTENT_BANDS"],
    ),
    (
        "extra shear bands",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}],
            all_is_shear_band=[True, False],
            all_wgts=[1],
            all_flags=[0, 0],
        ),
        ["INCONSISTENT_BANDS"],
        ["INCONSISTENT_BANDS", "PSF_FAILURE"],
        ["INCONSISTENT_BANDS"],
    ),
    (
        "flag a single shear",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[1, 0, 0, 0],
        ),
        ["NO_ATTEMPT"],
        ["NO_ATTEMPT", "PSF_FAILURE"],
        ["NO_ATTEMPT"],
    ),
    (
        "flag a shear res is fine",
        dict(
            all_res=[{
                "flux_flags": 1,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        ["NO_ATTEMPT"],
    ),
    (
        "zero weight a shear",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[0, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["ZERO_WEIGHTS"],
        ["ZERO_WEIGHTS", "PSF_FAILURE"],
        [],
    ),
    (
        "missing a shear res",
        dict(
            all_res=[None] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        ["MISSING_BAND"],
        ["MISSING_BAND"],
    ),
    (
        "zero weight a flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 0],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [],
    ),
    (
        "flag a flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 1],
        ),
        [],
        [],
        ["NO_ATTEMPT"],
    ),
    (
        "flag a flux in res",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3 + [
                {
                    "flux_flags": 1,
                    "flux": 1,
                    "flux_err": 1,
                    "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                    "mom_cov": np.diag(np.ones(6))
                }
            ],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        ["NO_ATTEMPT"],
    ),
    (
        "missing a flux res",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3 + [None],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        ["MISSING_BAND"],
    ),
    (
        "missing flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3 + [{
                "flux_flags": 0,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        ["NOMOMENTS_FAILURE"],
    ),
    (
        "missing flux_err",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3 + [{
                "flux_flags": 0,
                "flux": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        ["NOMOMENTS_FAILURE"],
    ),
    (
        "missing flux_flags",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3 + [{
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        ["NOMOMENTS_FAILURE"],
    ),
    (
        "missing mom",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom_cov": np.diag(np.ones(6))
            }] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        ["NOMOMENTS_FAILURE"],
        [],
    ),
    (
        "missing mom_cov",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1])
            }] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        ["NOMOMENTS_FAILURE"],
        [],
    ),
    (
        "missing psf mom",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {"mom_cov": np.diag(np.ones(6))}
            ] + [
                {"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}
            ] * 3,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["NOMOMENTS_FAILURE"],
        ["PSF_FAILURE"],
        [],
    ),
    (
        "missing psf mom cov",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {"mom": np.diag(np.ones(6))}
            ] + [
                {"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}
            ] * 3,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["NOMOMENTS_FAILURE"],
        ["PSF_FAILURE"],
        [],
    ),
    (
        "missing psf mom for flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}
            ] * 3 + [
                {"mom_cov": np.diag(np.ones(6))}
            ],
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [],
    ),
    (
        "missing psf mom for flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}
            ] * 3 + [
                {"mom": np.diag(np.ones(6))}
            ],
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [],
    ),
    (
        "negative/cancelling weights somehow",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 0.5, 0.5, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 5,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 5,
            all_is_shear_band=[True, True, True, False, False],
            all_wgts=[-1, 1, 0, 1, 1],
            all_flags=[0, 0, 0, 0, 0],
        ),
        ["ZERO_WEIGHTS"],
        ["ZERO_WEIGHTS", "PSF_FAILURE"],
        [],
    ),
    (
        "shear out of bounds",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                "mom": np.array([0, 0, 1, 1, 1, 1]),
                "mom_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[{"mom": np.ones(6), "mom_cov": np.diag(np.ones(6))}] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        ["SHEAR_RANGE_ERROR"],
        [],
    ),
])
def test_fitting_combine_fit_results_wavg_flagging(
    purpose, kwargs, psf_flags, model_flags, flux_flags
):
    def _print_flags(data):
        for name in data.dtype.names:
            if "flag" in name:
                print("    %s:" % name, procflags.get_procflags_str(data[name][0]))
        for name in data.dtype.names:
            if "flag" not in name:
                print("    %s:" % name, data[name][0])

    def _check_flags(val, flags):
        fval = 0
        for flag in flags:
            fval |= getattr(procflags, flag)
        if val != fval:
            for flag in flags:
                assert (val & getattr(procflags, flag)) != 0, (
                    "%s: flag val %s failed!" % (purpose, flag)
                )
        assert val == fval, purpose

    model = "wwmom"

    # all missing
    data = _combine_fit_results_wavg(model=model, **kwargs)
    print()
    _print_flags(data)
    _check_flags(data["psf_flags"][0], psf_flags)
    _check_flags(data[model + "_flags"][0], model_flags)
    _check_flags(data[model + "_band_flux_flags"][0], flux_flags)
    assert (
        data["flags"][0] ==
        (data[model + "_flags"][0] | data[model + "_band_flux_flags"][0])
    ), purpose
    if data["psf_flags"][0] != 0:
        assert (data[model + "_flags"][0] & procflags.PSF_FAILURE) != 0, purpose
