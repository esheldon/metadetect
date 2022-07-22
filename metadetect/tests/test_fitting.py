import numpy as np

import pytest

from ngmix.gaussmom import GaussMom
import ngmix

from .sim import make_mbobs_sim
from ..fitting import (
    fit_mbobs_wavg,
    _combine_fit_results_wavg,
    symmetrize_obs_weights,
    fit_all_psfs,
    _sum_bands_wavg,
    MOMNAME,
)
from .. import procflags


def _print_res(res):
    print("", flush=True)
    for name in res.dtype.names:
        if "flag" in name:
            if len(np.shape(res[name])) > 0 and np.shape(res[name])[0] > 1:
                for i, f in enumerate(res[name]):
                    print(
                        "    %s[%d]: %d (%s)" % (
                            name,
                            i,
                            res[name][i],
                            procflags.get_procflags_str(res[name][i]),
                        ),
                        flush=True,
                    )
            else:
                print(
                    "    %s: %d (%s)" % (
                        name,
                        res[name],
                        procflags.get_procflags_str(res[name]),
                    ),
                    flush=True,
                )
        else:
            print("    %s:" % name, res[name], flush=True)


def test_fit_all_psfs_same():
    mbobs1 = make_mbobs_sim(45, 4)
    fit_all_psfs(mbobs1, np.random.RandomState(seed=10))

    mbobs2 = make_mbobs_sim(45, 4)
    fit_all_psfs(mbobs2, np.random.RandomState(seed=10))

    for i in range(4):
        for key in mbobs1[i][0].psf.meta["result"]:
            assert np.all(
                mbobs1[i][0].psf.meta["result"][key]
                == mbobs2[i][0].psf.meta["result"][key]
            )


def test_fitting_fit_mbobs_wavg_flagging_nodata():
    mbobs = make_mbobs_sim(45, 4)
    mbobs[1] = ngmix.ObsList()
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    assert np.all((res["wmom_band_flux_flags"][:, 1] & procflags.MISSING_BAND) != 0)
    for i in [0, 2, 3]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
    mbobs[1] = ngmix.ObsList()
    shear_bands = [0, 1, 2, 3]
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        shear_bands=shear_bands,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["wmom_band_flux_flags"][:, 1] & procflags.MISSING_BAND) != 0)
    for i in [0, 2, 3, 4, 5, 6]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
    mbobs[5] = ngmix.ObsList()
    shear_bands = [0, 1, 2, 3]
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        shear_bands=shear_bands,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["wmom_band_flux_flags"][:, 5] & procflags.MISSING_BAND) != 0)
    for i in [0, 1, 2, 3, 4, 6]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 5]))
        for i in [0, 1, 2, 3, 4, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")


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
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    for f in [procflags.EDGE_HIT, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 1] & f) != 0)
    for i in [0, 2, 3]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
    with mbobs[1][0].writeable():
        mbobs[1][0].bmask[2, 3] = bmask_flags
        mbobs[1][0].bmask[3, 1] = other_flags
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        shear_bands=list(range(4)),
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    for f in [procflags.EDGE_HIT, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 1] & f) != 0)
    for i in [0, 2, 3, 4, 5, 6]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
    with mbobs[5][0].writeable():
        mbobs[5][0].bmask[2, 3] = bmask_flags
        mbobs[5][0].bmask[3, 1] = other_flags
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        shear_bands=list(range(4)),
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    for f in [procflags.EDGE_HIT, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 5] & f) != 0)
    for i in [0, 1, 2, 3, 4, 6]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 5]))
        for i in [0, 1, 2, 3, 4, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")


def test_fitting_fit_mbobs_wavg_flagging_zeroweight():
    mbobs = make_mbobs_sim(45, 4)
    with mbobs[1][0].writeable():
        mbobs[1][0].ignore_zero_weight = False
        mbobs[1][0].weight[:, :] = 0
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    for f in [procflags.ZERO_WEIGHTS, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 1] & f) != 0)
    for i in [0, 2, 3]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
    with mbobs[1][0].writeable():
        mbobs[1][0].ignore_zero_weight = False
        mbobs[1][0].weight[:, :] = 0
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        shear_bands=list(range(4)),
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    for f in [procflags.ZERO_WEIGHTS, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 1] & f) != 0)
    for i in [0, 2, 3, 4, 5, 6]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 1]))
        for i in [0, 2, 3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
    with mbobs[5][0].writeable():
        mbobs[5][0].ignore_zero_weight = False
        mbobs[5][0].weight[:, :] = 0
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=0,
        shear_bands=list(range(4)),
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    for f in [procflags.ZERO_WEIGHTS, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 5] & f) != 0)
    for i in [0, 1, 2, 3, 4, 6]:
        assert np.all(res["wmom_band_flux_flags"][:, i] == 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, 5]))
        for i in [0, 1, 2, 3, 4, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")


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
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all(
        (
            res["wmom_band_flux_flags"][:, 0]
            & (procflags.MISSING_BAND)
        ) != 0
    )
    for f in [procflags.EDGE_HIT, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 1] & f) != 0)
    for f in [procflags.ZERO_WEIGHTS, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 2] & f) != 0)
    assert np.all(res["wmom_band_flux_flags"][:, 3:] == 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, 3]))
        for i in [0, 1, 2]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
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
        shear_bands=list(range(4)),
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all(
        (
            res["wmom_band_flux_flags"][:, 0]
            & (procflags.MISSING_BAND)
        ) != 0
    )
    for f in [procflags.EDGE_HIT, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 1] & f) != 0)
    for f in [procflags.ZERO_WEIGHTS, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 2] & f) != 0)
    assert np.all(res["wmom_band_flux_flags"][:, 3:] == 0)
    assert not np.any(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        for i in [3, 4, 5, 6]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
        for i in [0, 1, 2]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")

    mbobs = make_mbobs_sim(45, 7)
    bmask_flags = 2**8
    other_flags = 2**3
    mbobs[4] = ngmix.ObsList()
    with mbobs[5][0].writeable():
        mbobs[5][0].ignore_zero_weight = False
        mbobs[5][0].weight[:, :] = 0
    with mbobs[6][0].writeable():
        mbobs[6][0].bmask[2, 3] = bmask_flags
        mbobs[6][0].bmask[3, 1] = other_flags
    res = fit_mbobs_wavg(
        mbobs=mbobs,
        fitter=GaussMom(1.2),
        bmask_flags=bmask_flags,
        shear_bands=list(range(4)),
    )
    _print_res(res[0])
    assert np.all((res["flags"] & procflags.MISSING_BAND) != 0)
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all(
        (
            res["wmom_band_flux_flags"][:, 4]
            & (procflags.MISSING_BAND)
        ) != 0
    )
    for f in [procflags.ZERO_WEIGHTS, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 5] & f) != 0)
    for f in [procflags.EDGE_HIT, procflags.MISSING_BAND]:
        assert np.all((res["wmom_band_flux_flags"][:, 6] & f) != 0)
    assert np.all(res["wmom_band_flux_flags"][:, :4] == 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        for i in [0, 1, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
        for i in [4, 5, 6]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))
    assert np.all(res["shear_bands"] == "0123")


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
        [["MISSING_BAND"]] * 4,
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
        [
            ["MISSING_BAND", "NO_ATTEMPT"],
            ["MISSING_BAND"],
            ["MISSING_BAND", "NO_ATTEMPT"],
            ["MISSING_BAND"]
        ],
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
        [["MISSING_BAND"]] * 4,
    ),
    (
        "everything is fine one band",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 1,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 1,
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
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[]] * 4,
    ),
    (
        "extra shear bands",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 2,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 1,
            all_is_shear_band=[True],
            all_wgts=[1],
            all_flags=[0],
        ),
        ["INCONSISTENT_BANDS"],
        ["INCONSISTENT_BANDS", "PSF_FAILURE"],
        [["INCONSISTENT_BANDS"], ["INCONSISTENT_BANDS"]],
    ),
    (
        "extra PSF bands",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 2,
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
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}],
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
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}],
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
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[{MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}],
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
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[1, 0, 0, 0],
        ),
        ["NO_ATTEMPT"],
        ["NO_ATTEMPT", "PSF_FAILURE"],
        [["NO_ATTEMPT"], [], [], []],
    ),
    (
        "flag a shear res is fine",
        dict(
            all_res=[{
                "flux_flags": 1,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [["NO_ATTEMPT"], [], [], []],
    ),
    (
        "zero weight a shear",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[0, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["ZERO_WEIGHTS"],
        ["ZERO_WEIGHTS", "PSF_FAILURE"],
        [[]] * 4,
    ),
    (
        "missing a shear res",
        dict(
            all_res=[None] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["MISSING_BAND"],
        ["MISSING_BAND", "PSF_FAILURE"],
        [["MISSING_BAND"], [], [], []],
    ),
    (
        "zero weight a flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 0],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[]] * 4,
    ),
    (
        "flag a flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 1],
        ),
        [],
        [],
        [[], [], [], ["NO_ATTEMPT"]],
    ),
    (
        "flag a flux in res",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3 + [
                {
                    "flux_flags": 1,
                    "flux": 1,
                    "flux_err": 1,
                    MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                    MOMNAME+"_cov": np.diag(np.ones(6))
                }
            ],
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[], [], [], ["NO_ATTEMPT"]],
    ),
    (
        "missing a flux res",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3 + [None],
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[], [], [], ["MISSING_BAND"]],
    ),
    (
        "missing flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3 + [{
                "flux_flags": 0,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[], [], [], ["NOMOMENTS_FAILURE"]],
    ),
    (
        "missing flux_err",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3 + [{
                "flux_flags": 0,
                "flux": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[], [], [], ["NOMOMENTS_FAILURE"]],
    ),
    (
        "missing flux_flags",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3 + [{
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }],
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[], [], [], ["NOMOMENTS_FAILURE"]],
    ),
    (
        "missing mom",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["NOMOMENTS_FAILURE"],
        ["NOMOMENTS_FAILURE", "PSF_FAILURE"],
        [[]] * 4,
    ),
    (
        "missing mom_cov",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1])
            }] + [{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 3,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 4,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["NOMOMENTS_FAILURE"],
        ["NOMOMENTS_FAILURE", "PSF_FAILURE"],
        [[]] * 4,
    ),
    (
        "missing psf mom",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME+"_cov": np.diag(np.ones(6))}
            ] + [
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 3,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["NOMOMENTS_FAILURE"],
        ["PSF_FAILURE"],
        [[]] * 4,
    ),
    (
        "missing psf mom cov",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6)}
            ] + [
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 3,
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        ["NOMOMENTS_FAILURE"],
        ["PSF_FAILURE"],
        [[]] * 4,
    ),
    (
        "missing psf mom for flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 3 + [
                {MOMNAME+"_cov": np.diag(np.ones(6))}
            ],
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[]] * 4,
    ),
    (
        "missing psf mom for flux",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 4,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 3 + [
                {MOMNAME: np.diag(np.ones(6))}
            ],
            all_is_shear_band=[True, True, False, False],
            all_wgts=[1, 1, 1, 1],
            all_flags=[0, 0, 0, 0],
        ),
        [],
        [],
        [[]] * 4,
    ),
    (
        "negative/cancelling weights somehow",
        dict(
            all_res=[{
                "flux_flags": 0,
                "flux": 1,
                "flux_err": 1,
                MOMNAME: np.array([0, 0, 0.5, 0.5, 1, 1]),
                MOMNAME+"_cov": np.diag(np.ones(6))
            }] * 5,
            all_psf_res=[
                {MOMNAME: np.ones(6), MOMNAME+"_cov": np.diag(np.ones(6))}
            ] * 5,
            all_is_shear_band=[True, True, True, False, False],
            all_wgts=[-1, 1, 0, 1, 1],
            all_flags=[0, 0, 0, 0, 0],
        ),
        ["ZERO_WEIGHTS"],
        ["ZERO_WEIGHTS", "PSF_FAILURE"],
        [[]] * 5,
    ),
])
def test_fitting_combine_fit_results_wavg_flagging(
    purpose, kwargs, psf_flags, model_flags, flux_flags
):
    def _check_flags(val, flags):
        if np.shape(val) == tuple():
            fval = 0
            for flag in flags:
                fval |= getattr(procflags, flag)
            if val != fval:
                for flag in flags:
                    assert (val & getattr(procflags, flag)) != 0, (
                        "%s: flag val %s failed!" % (purpose, flag)
                    )
            assert val == fval, purpose
        else:
            for i, _val in enumerate(val):
                fval = 0
                for flag in flags[i]:
                    fval |= getattr(procflags, flag)

                if _val != fval:
                    for flag in flags:
                        assert (_val & getattr(procflags, flag)) != 0, (
                            "%s: flag val %s failed!" % (purpose, flag)
                        )
                assert _val == fval, purpose

    model = "wwmom"
    shear_bands = [i for i, b in enumerate(kwargs["all_is_shear_band"]) if b]

    data = _combine_fit_results_wavg(model=model, shear_bands=shear_bands, **kwargs)
    print()
    _print_res(data[0])
    _check_flags(data["psf_flags"][0], psf_flags)
    _check_flags(data[model + "_flags"][0], model_flags)
    _check_flags(data[model + "_band_flux_flags"][0], flux_flags)
    if len(flux_flags) > 1 and isinstance(flux_flags[0], list):
        dff = 0
        for f in data[model + "_band_flux_flags"][0]:
            dff |= f
    else:
        dff = data[model + "_band_flux_flags"][0]
    assert (
        data["flags"][0] ==
        (data[model + "_flags"][0] | dff)
    ), purpose
    if data["psf_flags"][0] != 0:
        assert (data[model + "_flags"][0] & procflags.PSF_FAILURE) != 0, purpose


@pytest.mark.parametrize("mom_norm", [None, [0.3, 0.9, 0.8, 0.6]])
def test_fitting_sum_bands_wavg_weighting(mom_norm):
    all_is_shear_band = [True, True]
    all_res = [
        {
            MOMNAME: np.ones(6) * 3,
            MOMNAME+"_cov": np.diag(np.ones(6)) * 3.1,
        },
        {
            MOMNAME: np.ones(6) * 7,
            MOMNAME+"_cov": np.diag(np.ones(6)) * 7.1,
        },
    ]
    all_wgts = [0.2, 0.5]
    all_flags = [0, 0]
    all_wgt_res = [
        {
            MOMNAME: np.ones(6) * 6,
            MOMNAME + "_cov": np.diag(np.ones(6)) * 6.1,
        },
        {
            MOMNAME: np.ones(6) * 2,
            MOMNAME+"_cov": np.diag(np.ones(6)) * 2.1,
        },
    ]

    if mom_norm is not None:
        all_res[0][MOMNAME+"_norm"] = mom_norm[0]
        all_res[1][MOMNAME+"_norm"] = mom_norm[1]
        all_wgt_res[0][MOMNAME+"_norm"] = mom_norm[2]
        all_wgt_res[1][MOMNAME+"_norm"] = mom_norm[3]

    # return value is
    # raw_mom, raw_mom_cov, wgt_sum, final_flags, used_shear_bands,
    # flux, flux_var, flux_wgt_sum
    sums_wgt = _sum_bands_wavg(
        all_res=all_res,
        all_is_shear_band=all_is_shear_band,
        all_wgts=all_wgts,
        all_flags=all_flags,
        all_wgt_res=all_wgt_res,
    )
    if mom_norm is None:
        fac0 = 6/3
        fac1 = 2/7
    else:
        fac0 = (6 / mom_norm[2]) / (3 / mom_norm[0])
        fac1 = (2 / mom_norm[3]) / (7 / mom_norm[1])

    assert sums_wgt["wgt_sum"] == 0.7
    if mom_norm is None:
        np.testing.assert_allclose(
            sums_wgt["raw_mom"],
            [0.2 * 3 * fac0 + 0.5 * 7 * fac1] * 6
        )
    else:
        np.testing.assert_allclose(
            sums_wgt["raw_mom"],
            [0.2 * 3 * fac0 / mom_norm[0] + 0.5 * 7 * fac1 / mom_norm[1]] * 6
        )
    if mom_norm is None:
        np.testing.assert_allclose(
            np.diag(sums_wgt["raw_mom_cov"]),
            [0.2**2 * 3.1 * fac0**2 + 0.5**2 * 7.1 * fac1**2] * 6
        )
    else:
        np.testing.assert_allclose(
            np.diag(sums_wgt["raw_mom_cov"]),
            [
                0.2**2 * 3.1 * fac0**2 / mom_norm[0]**2
                + 0.5**2 * 7.1 * fac1**2 / mom_norm[1]**2
            ] * 6
        )

    sums = _sum_bands_wavg(
        all_res=all_res,
        all_is_shear_band=all_is_shear_band,
        all_wgts=all_wgts,
        all_flags=all_flags,
        all_wgt_res=None,
    )
    assert sums["wgt_sum"] == 0.7
    if mom_norm is None:
        np.testing.assert_allclose(
            sums["raw_mom"],
            [0.2 * 3 + 0.5 * 7] * 6
        )
    else:
        np.testing.assert_allclose(
            sums["raw_mom"],
            [0.2 * 3 / mom_norm[0] + 0.5 * 7 / mom_norm[1]] * 6
        )
    if mom_norm is None:
        np.testing.assert_allclose(
            np.diag(sums["raw_mom_cov"]),
            [0.2**2 * 3.1 + 0.5**2 * 7.1] * 6
        )
    else:
        np.testing.assert_allclose(
            np.diag(sums["raw_mom_cov"]),
            [0.2**2 * 3.1 / mom_norm[0]**2 + 0.5**2 * 7.1 / mom_norm[1]**2] * 6
        )

    # everything but the moments should be the same
    for key in [
        "wgt_sum", "final_flags", "used_shear_bands", "flux", "flux_var",
    ]:
        assert sums[key] == sums_wgt[key], (key, sums[key])


def test_fitting_symmetrize_obs_weights_all_zero():
    obs = ngmix.Observation(
        image=np.zeros((13, 13)),
        weight=np.zeros((13, 13)),
        ignore_zero_weight=False,
    )
    sym_obs = symmetrize_obs_weights(obs)
    assert sym_obs is not obs
    assert sym_obs.ignore_zero_weight is False
    assert np.all(sym_obs.weight == 0)

    wgt = np.zeros((13, 13))
    wgt[:, :2] = 1
    obs = ngmix.Observation(
        image=np.zeros((13, 13)),
        weight=wgt,
    )
    sym_obs = symmetrize_obs_weights(obs)
    assert sym_obs is not obs
    assert sym_obs.ignore_zero_weight is False
    assert np.all(sym_obs.weight == 0)
    assert not np.array_equal(sym_obs.weight, obs.weight)


def test_fitting_symmetrize_obs_weights_none():
    obs = ngmix.Observation(
        image=np.zeros((13, 13)),
        weight=np.ones((13, 13)),
    )
    sym_obs = symmetrize_obs_weights(obs)
    assert sym_obs is not obs
    assert sym_obs.ignore_zero_weight is True
    assert np.all(sym_obs.weight == 1)
    assert np.array_equal(sym_obs.weight, obs.weight)


def test_fitting_symmetrize_obs_weights():
    wgt = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 1],
    ])
    obs = ngmix.Observation(
        image=np.zeros((3, 3)),
        weight=wgt,
    )
    sym_obs = symmetrize_obs_weights(obs)
    assert sym_obs is not obs
    assert sym_obs.ignore_zero_weight is True
    assert not np.array_equal(sym_obs.weight, obs.weight)
    sym_wgt = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ])
    assert np.array_equal(sym_obs.weight, sym_wgt)


def test_fitting_fit_mbobs_wavg_wmom_tratio():
    fitter = GaussMom(1.2)
    seed = 10
    nband = 3
    bmask_flags = 0

    mbobs = make_mbobs_sim(
        seed,
        nband,
        simulate_star=True,
        noise_scale=1e-4,
        band_flux_factors=[0.1, 2.0, 5.0],
        band_image_sizes=[39, 45, 67],
    )
    res = fit_mbobs_wavg(mbobs=mbobs, fitter=fitter, bmask_flags=bmask_flags)
    _print_res(res[0])
    assert np.allclose(res["wmom_T_ratio"], 1.0)

    mbobs = make_mbobs_sim(
        seed,
        nband,
        simulate_star=False,
        noise_scale=1e-4,
        band_flux_factors=[0.1, 2.0, 5.0],
        band_image_sizes=[39, 45, 67],
    )
    res = fit_mbobs_wavg(mbobs=mbobs, fitter=fitter, bmask_flags=bmask_flags)
    _print_res(res[0])
    assert not np.allclose(res["wmom_T_ratio"], 1.0)
    assert res["wmom_T_ratio"][0] > 1.5
