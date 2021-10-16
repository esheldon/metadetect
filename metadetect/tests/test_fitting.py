import numpy as np

from ngmix.gaussmom import GaussMom
import ngmix

from .sim import make_mbobs_sim
from ..fitting import fit_mbobs_wavg
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
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
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        for i in [0, 1, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
        for i in [4, 5, 6]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))
