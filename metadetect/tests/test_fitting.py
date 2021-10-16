import numpy as np
import pytest

from ngmix.prepsfmom import PrePSFGaussMom
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
    assert np.all((res["flags"] & procflags.NO_DATA) != 0)
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
    assert np.all((res["flags"] & procflags.NO_DATA) != 0)
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
    assert np.all((res["flags"] & procflags.NO_DATA) != 0)
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
    assert np.all((res["flags"] & procflags.NO_DATA) != 0)
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
    assert np.all((res["flags"] & procflags.NO_DATA) != 0)
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
    assert np.all((res["flags"] & procflags.NO_DATA) != 0)
    assert np.all((res["flags"] & procflags.ZERO_WEIGHTS) != 0)
    assert np.all((res["flags"] & procflags.EDGE_HIT) != 0)
    assert np.all((res["wmom_band_flux_flags"] & procflags.OBJ_FAILURE) != 0)
    assert np.all(np.isfinite(res["wmom_g_cov"]))
    for tail in ["", "_err"]:
        for i in [0, 1, 2, 3]:
            assert np.all(np.isfinite(res["wmom_band_flux" + tail][:, i]))
        for i in [4, 5, 6]:
            assert not np.any(np.isfinite(res["wmom_band_flux" + tail][:, i]))


# @pytest.mark.parametrize("fitter,model", [(Moments, "wmom"), (KSigmaMoments, "ksigma")])
# @pytest.mark.parametrize("nobj", [1, 2, 11])
# def test_metadetect_wavg_comp_single_band(nobj, fitter, model):
#     """test that computing the weighted averages with one band gives the
#     same result as the inputs.
#     """
#     if KSigmaMoments is None and model == "ksigma":
#         pytest.skip()
#
#     # sim the mbobs list
#     mbobs_list = make_mbobs_sim(134341, nobj, 1)[0]
#     momres = fitter(
#         {"weight": {"fwhm": 1.2}, "bmask_flags": 0},
#         rng=np.random.RandomState(seed=12),
#     ).go(mbobs_list)
#
#     # now we make an Metadetect object
#     # note we are making a sim here but not using it
#     sim = Sim(np.random.RandomState(seed=329058), config={"nband": 1})
#     config = {}
#     config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
#     config["model"] = model
#     sim_mbobs = sim.get_mbobs()
#     mdet = metadetect.Metadetect(config, sim_mbobs, np.random.RandomState(seed=14328))
#
#     wgts = np.array([1])
#     all_is_shear_band = [True]
#     any_nonzero = False
#     for i, mbobs in enumerate(mbobs_list):
#         all_bres = [momres[i:i+1]]
#         res = mdet._compute_wavg_fitter_mbobs_sep(
#             wgts, all_bres, all_is_shear_band, mbobs
#         )
#         for col in [
#             f"{model}_T", f"{model}_T_err", f"{model}_g", f"{model}_g_cov",
#             f"{model}_s2n", "flags", f"{model}_T_ratio", f"{model}_flags",
#             "psf_T", "psf_g",
#         ]:
#             if np.any(res[col] > 0):
#                 any_nonzero = True
#             assert np.allclose(res[col], momres[col][i]), col
#
#     assert any_nonzero
#
#
# @pytest.mark.parametrize("fitter,model", [(Moments, "wmom"), (KSigmaMoments, "ksigma")])
# @pytest.mark.parametrize("nband", [2, 3, 4])
# @pytest.mark.parametrize("nobj", [1, 2, 11])
# def test_metadetect_wavg_comp(nband, nobj, fitter, model):
#     """test that the weighted averages for shear are computed correctly."""
#     if KSigmaMoments is None and model == "ksigma":
#         pytest.skip()
#
#     # sim the mbobs list
#     band_mbobs_list = make_mbobs_sim(134341, nobj, nband)
#     band_momres = [
#         fitter(
#             {"weight": {"fwhm": 1.2}, "bmask_flags": 0},
#             rng=np.random.RandomState(seed=12),
#         ).go(mbobs_list)
#         for mbobs_list in band_mbobs_list
#     ]
#
#     # now we make an Metadetect object
#     # note we are making a sim here but not using it
#     sim = Sim(np.random.RandomState(seed=329058), config={"nband": nband})
#     config = {}
#     config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
#     config["model"] = model
#     sim_mbobs = sim.get_mbobs()
#     mdet = metadetect.Metadetect(config, sim_mbobs, np.random.RandomState(seed=14328))
#
#     all_is_shear_band = [True] * nband
#     any_nonzero = False
#     for i in range(nobj):
#         shear_mbobs = ngmix.MultiBandObsList()
#         for band in range(nband):
#             shear_mbobs.append(band_mbobs_list[band][i][0])
#         all_bres = [momres[i:i+1] for momres in band_momres]
#         wgts = np.array(
#             [band_mbobs_list[b][i][0][0].meta["wgt"] for b in range(nband)]
#         )
#         wgts /= np.sum(wgts)
#         res = mdet._compute_wavg_fitter_mbobs_sep(
#             wgts, all_bres, all_is_shear_band,
#             shear_mbobs,
#         )
#         # check a subset and don't go crazy
#         for col in [
#             "flags", f"{model}_flags", "psf_T", "psf_g",
#             f"{model}_band_flux", f"{model}_band_flux_err",
#             f"{model}_s2n", f"{model}_g", f"{model}_T",
#         ]:
#             if np.any(res[col] > 0):
#                 any_nonzero = True
#
#             if col in ["psf_T", "psf_g"]:
#                 val = np.sum([
#                     wgt * momres[col][i:i+1] for wgt, momres in zip(wgts, band_momres)
#                 ], axis=0)
#             elif col in ["flags", f"{model}_flags"]:
#                 val = 0
#                 for momres in band_momres:
#                     val |= momres[col][i:i+1]
#             elif col in [f"{model}_band_flux", f"{model}_band_flux_err"]:
#                 val = np.array([
#                     momres[col.replace("band_", "")][i:i+1]
#                     for momres in band_momres
#                 ]).T
#             elif col in [f"{model}_T"]:
#                 val = np.sum([
#                     wgt * momres[f"{model}_raw_mom"][i:i+1, 1]
#                     for wgt, momres in zip(wgts, band_momres)
#                 ], axis=0)
#                 val /= np.sum([
#                     wgt * momres[f"{model}_raw_mom"][i:i+1, 0]
#                     for wgt, momres in zip(wgts, band_momres)
#                 ], axis=0)
#             elif col in [f"{model}_s2n"]:
#                 val = np.sum([
#                     wgt * momres[f"{model}_raw_mom"][i, 0]
#                     for wgt, momres in zip(wgts, band_momres)
#                 ])
#                 val /= np.sqrt(np.sum([
#                     wgt**2 * momres[f"{model}_raw_mom_cov"][i, 0, 0]
#                     for wgt, momres in zip(wgts, band_momres)
#                 ]))
#             elif col in [f"{model}_g"]:
#                 val = np.sum([
#                     wgt * momres[f"{model}_raw_mom"][i:i+1, 2:]
#                     for wgt, momres in zip(wgts, band_momres)
#                 ], axis=0)
#                 val /= np.sum([
#                     wgt * momres[f"{model}_raw_mom"][i:i+1, 1]
#                     for wgt, momres in zip(wgts, band_momres)
#                 ], axis=0)
#             else:
#                 assert False, "col %s not in elif block for test!" % col
#
#             assert np.allclose(res[col], val), col
#
#     assert any_nonzero
#
#
# @pytest.mark.parametrize("fitter,model", [(Moments, "wmom"), (KSigmaMoments, "ksigma")])
# def test_metadetect_wavg_flagging(fitter, model):
#     """test that the weighted averages for shear are computed correctly."""
#     if KSigmaMoments is None and model == "ksigma":
#         pytest.skip()
#
#     # sim the mbobs list
#     nband = 2
#     nobj = 4
#     band_mbobs_list = make_mbobs_sim(134341, nobj, nband)
#     band_momres = [
#         fitter(
#             {"weight": {"fwhm": 1.2}, "bmask_flags": 0},
#             rng=np.random.RandomState(seed=12),
#         ).go(mbobs_list)
#         for mbobs_list in band_mbobs_list
#     ]
#
#     # now we make an Metadetect object
#     # note we are making a sim here but not using it
#     sim = Sim(np.random.RandomState(seed=329058), config={"nband": nband})
#     config = {}
#     config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
#     config["model"] = model
#     sim_mbobs = sim.get_mbobs()
#     mdet = metadetect.Metadetect(config, sim_mbobs, np.random.RandomState(seed=14328))
#
#     all_is_shear_band = [True] * nband
#     for i in range(nobj):
#         shear_mbobs = ngmix.MultiBandObsList()
#         for band in range(nband):
#             shear_mbobs.append(band_mbobs_list[band][i][0])
#         all_bres = [momres[i:i+1] for momres in band_momres]
#         wgts = np.array(
#             [band_mbobs_list[b][i][0][0].meta["wgt"] for b in range(nband)]
#         )
#         wgts /= np.sum(wgts)
#
#         nonshear_mbobs = None
#         if i == 0:
#             shear_mbobs[1] = ngmix.ObsList()
#         elif i == 1:
#             wgts[0] = 0.0
#         elif i == 2:
#             nonshear_mbobs = ngmix.MultiBandObsList()
#             nonshear_mbobs.append(ngmix.ObsList())
#
#         res = mdet._compute_wavg_fitter_mbobs_sep(
#             wgts, all_bres, all_is_shear_band,
#             shear_mbobs, nonshear_mbobs=nonshear_mbobs,
#         )
#
#         if i in [0, 1, 2]:
#             assert (res["flags"] & procflags.OBJ_FAILURE) != 0
#             assert (res[f"{model}_flags"] & procflags.OBJ_FAILURE) != 0
