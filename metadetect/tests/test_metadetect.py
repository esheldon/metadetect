"""
test with super simple sim.  The purpose here is not
to make sure it gets the right answer or anything, just
to test all the moving parts
"""
import time
import copy
import itertools

import pytest

from packaging import version
import ngmix
import numpy as np

from .. import detect
from .. import metadetect
from .. import fitting
from .. import procflags
from .sim import Sim


TEST_METADETECT_CONFIG = {
    "model": "wmom",

    "weight": {
        "fwhm": 1.2,  # arcsec
    },

    "metacal": {
        "psf": "fitgauss",
        "types": ["noshear", "1p", "1m", "2p", "2m"],
    },

    "sx": {
        # in sky sigma
        # DETECT_THRESH
        "detect_thresh": 0.8,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        "deblend_cont": 0.00001,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        "minarea": 4,

        "filter_type": "conv",

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        "filter_kernel": [
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
        ]
    },

    "meds": {
        "min_box_size": 32,
        "max_box_size": 256,

        "box_type": "iso_radius",

        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },

    # check for an edge hit
    "bmask_flags": 2**30,

    "nodet_flags": 2**0,
}


def _show_mbobs(mer):
    import images

    mbobs = mer.mbobs

    rgb = images.get_color_image(
        mbobs[2][0].image.transpose(),
        mbobs[1][0].image.transpose(),
        mbobs[0][0].image.transpose(),
        nonlinear=0.1,
    )
    rgb *= 1.0/rgb.max()

    images.view_mosaic(
        [rgb,
         mer.seg,
         mer.detim],
        titles=["image", "seg", "detim"],
    )


def test_detect(ntrial=1, show=False):
    """
    just test the detection
    """
    pytest.importorskip("sxdes")

    rng = np.random.RandomState(seed=45)

    tm0 = time.time()
    nobj_meas = 0

    sim = Sim(rng)

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        mer = detect.MEDSifier(
            mbobs=mbobs,
            sx_config=config["sx"],
            meds_config=config["meds"],
        )

        mbm = mer.get_multiband_meds()

        nobj = mbm.size
        nobj_meas += nobj

        if show:
            _show_mbobs(mer)
            if ntrial > 1 and trial != (ntrial-1):
                if "q" == input("hit a key: "):
                    return

    total_time = time.time()-tm0
    print("found", nobj_meas, "objects")
    print("time per group:", total_time/ntrial)
    print("time per object:", total_time/nobj_meas)


def test_detect_masking(ntrial=1, show=False):
    """
    just test the detection
    """
    pytest.importorskip("sxdes")

    rng = np.random.RandomState(seed=45)

    sim = Sim(rng)

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for obslist in mbobs:
            for obs in obslist:
                obs.bmask = obs.bmask | config["nodet_flags"]

        mer = detect.MEDSifier(
            mbobs=mbobs,
            sx_config=config["sx"],
            meds_config=config["meds"],
            nodet_flags=config["nodet_flags"],
        )
        assert mer.cat.size == 0


def _check_result_array(res, shear, msk, model):
    for col in res[shear].dtype.names:
        if col == "shear_bands":
            assert np.all(res[shear][msk][col] == "012")
        elif col == "det_bands":
            assert np.all(res[shear][msk][col] == "012")
        else:
            # admom doesn't make band fluxes
            if model in ["admom", "am", "gauss"] and "band_flux" in col:
                if col.endswith("band_flux_flags"):
                    assert np.array_equal(
                        res[shear][msk][col],
                        np.zeros_like(res[shear][msk][col])
                        + procflags.NO_ATTEMPT,
                    ), (
                        "result column '%s' is not NO_ATTEMPT: %s" % (
                            col, res[shear][msk][col]
                        )
                    )
                elif any(
                    col.endswith(s) for s in ["band_flux", "band_flux_err"]
                ):
                    assert np.all(np.isnan(
                        res[shear][msk][col],
                    )), (
                        "result column '%s' is not NaN: %s" % (
                            col, res[shear][msk][col]
                        )
                    )
            else:
                assert np.all(np.isfinite(res[shear][msk][col])), (
                    "result column '%s' has NaNs: %s" % (
                        col, res[shear][msk][col]
                    )
                )


@pytest.mark.parametrize("model", ["gauss", "wmom"])
def test_metadetect_coadd_faster(model):
    """
    test coadding is faster
    """
    pytest.importorskip("sxdes")

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    del config["model"]
    del config["weight"]
    config["fitters"] = [
        {"model": model, "coadd": False, "weight": {"fwhm": 1.2}}
    ]

    # warm up once
    mbobs = Sim(
        np.random.RandomState(seed=116), config={"nband": 6}
    ).get_mbobs()
    metadetect.do_metadetect(
        config, mbobs, np.random.RandomState(seed=116)
    )

    mbobs = Sim(
        np.random.RandomState(seed=116), config={"nband": 6}
    ).get_mbobs()
    tm0 = time.time()
    metadetect.do_metadetect(
        config, mbobs, np.random.RandomState(seed=116)
    )
    no_coadd_time = time.time() - tm0

    config["fitters"][0]["coadd"] = True
    mbobs = Sim(
        np.random.RandomState(seed=116), config={"nband": 6}
    ).get_mbobs()
    tm0 = time.time()
    metadetect.do_metadetect(
        config, mbobs, np.random.RandomState(seed=116)
    )
    coadd_time = time.time() - tm0

    print("coadd|nocoadd: %f|%f" % (coadd_time, no_coadd_time))

    if model == "gauss":
        assert coadd_time < no_coadd_time*0.7, (coadd_time, no_coadd_time)
    else:
        assert np.allclose(coadd_time, no_coadd_time, atol=0, rtol=0.3)


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_smoke(model):
    """
    test full metadetection
    """
    pytest.importorskip("sxdes")

    ntrial = 1
    rng = np.random.RandomState(seed=116)

    tm0 = time.time()

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        res = metadetect.do_metadetect(config, mbobs, rng)
        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.all(res[shear]["mfrac"] == 0)
            assert any(c.endswith("band_flux") for c in res[shear].dtype.names)
            assert np.any(res[shear]["psfrec_g"] != 0)
            assert np.any(res[shear]["psfrec_T"] != 0)
            msk = res[shear][model + '_flags'] == 0
            _check_result_array(res, shear, msk, model)

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_uberseg(model):
    """
    test full metadetection
    """
    pytest.importorskip("sxdes")

    ntrial = 1
    rng = np.random.RandomState(seed=116)

    tm0 = time.time()

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model
    config["meds"]["weight_type"] = "uberseg"

    mbobs = sim.get_mbobs()
    res = metadetect.do_metadetect(config, mbobs, rng)

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        res = metadetect.do_metadetect(config, mbobs, rng)
        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.all(res[shear]["mfrac"] == 0)
            assert any(c.endswith("band_flux") for c in res[shear].dtype.names)
            assert np.any(res[shear]["psfrec_g"] != 0)
            assert np.any(res[shear]["psfrec_T"] != 0)
            msk = res[shear][model + '_flags'] == 0
            _check_result_array(res, shear, msk, model)

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_mfrac(model):
    """
    test full metadetection w/ mfrac
    """
    pytest.importorskip("sxdes")

    ntrial = 1
    rng = np.random.RandomState(seed=53341)

    tm0 = time.time()

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for band in range(len(mbobs)):
            mbobs[band][0].mfrac = rng.uniform(
                size=mbobs[band][0].image.shape, low=0.2, high=0.8
            )
        res = metadetect.do_metadetect(config, mbobs, rng)
        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.all(
                (res[shear]["mfrac"] > 0.45)
                & (res[shear]["mfrac"] < 0.55)
            )
            assert np.all(
                (res[shear]["mfrac_img"] > 0.45)
                & (res[shear]["mfrac_img"] < 0.55)
            )
            assert np.all(
                (res[shear]["mfrac_noshear"] > 0.45)
                & (res[shear]["mfrac_noshear"] < 0.55)
            )
            msk = res[shear][model + '_flags'] == 0
            _check_result_array(res, shear, msk, model)

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_mfrac_all(model):
    """
    test full metadetection w/ mfrac all 1
    """
    ntrial = 1
    rng = np.random.RandomState(seed=53341)

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for band in range(len(mbobs)):
            mbobs[band][0].mfrac = np.ones_like(mbobs[band][0].image)

        res = metadetect.do_metadetect(config, mbobs, rng)
        assert res is None


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_zero_weight_all(model):
    """
    test full metadetection w/ all zero weight
    """

    ntrial = 1
    rng = np.random.RandomState(seed=53341)

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for band in range(len(mbobs)):
            mbobs[band][0].weight = np.zeros_like(mbobs[band][0].image)

        res = metadetect.do_metadetect(config, mbobs, rng)
        assert res is None


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_zero_weight_some(model):
    """
    test full metadetection w/ some zero weight
    """
    ntrial = 1
    rng = np.random.RandomState(seed=53341)

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for band in range(len(mbobs)):
            if band == 1:
                mbobs[band][0].weight = np.zeros_like(mbobs[band][0].image)

        res = metadetect.do_metadetect(config, mbobs, rng)
        assert res is None


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_nodet_flags_all(model):
    """
    test full metadetection w/ all bmask all nodet_flags
    """
    ntrial = 1
    rng = np.random.RandomState(seed=53341)

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for band in range(len(mbobs)):
            mbobs[band][0].bmask = np.ones_like(
                mbobs[band][0].image, dtype=np.int32
            )

        res = metadetect.do_metadetect(config, mbobs, rng)
        assert res is None


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma", "am", "gauss"])
def test_metadetect_nodet_flags_some(model):
    """
    test full metadetection w/ some bmask nodet_flags
    """
    ntrial = 1
    rng = np.random.RandomState(seed=53341)

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for band in range(len(mbobs)):
            if band == 1:
                mbobs[band][0].bmask = np.ones_like(
                    mbobs[band][0].image, dtype=np.int32
                )

        res = metadetect.do_metadetect(config, mbobs, rng)
        assert res is None


@pytest.mark.skipif(
    version.parse(ngmix.__version__) < version.parse("2.1.0"),
    reason="ngmix version 2.1.0 or greater is needed for smoothing prepsf moments",
)
@pytest.mark.parametrize("model", ["pgauss", "ksigma"])
def test_metadetect_fitter_fwhm_smooth(model):
    pytest.importorskip("sxdes")

    nband = 3
    rng = np.random.RandomState(seed=116)

    sim = Sim(rng, config={"nband": nband})
    mbobs = sim.get_mbobs()

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model
    config["weight"]["fwhm"] = 1.2

    md = metadetect.Metadetect(
        config, mbobs, rng,
    )
    md.go()
    res = md.result
    assert md._fitters[0].fwhm_smooth == 0

    config["weight"]["fwhm_smooth"] = 0.8
    md = metadetect.Metadetect(
        config, mbobs, rng,
    )
    md.go()
    res_smooth = md.result
    assert md._fitters[0].fwhm_smooth == 0.8

    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        msk = res[shear][model + "_flags"] == 0
        msk_smooth = res_smooth[shear][model + "_flags"] == 0
        assert (
            np.mean(res_smooth[shear][model + "_T"][msk_smooth])
            > np.mean(res[shear][model + "_T"][msk])
        )
        assert (
            np.mean(res_smooth[shear][model + "_g_cov"][msk_smooth, 0, 0])
            < np.mean(res[shear][model + "_g_cov"][msk, 0, 0])
        )


@pytest.mark.parametrize("model", ["pgauss", "ksigma"])
def test_metadetect_fitter_fwhm_reg(model):
    pytest.importorskip("sxdes")

    nband = 3
    rng = np.random.RandomState(seed=116)

    sim = Sim(rng, config={"nband": nband})
    mbobs = sim.get_mbobs()

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model
    config["weight"]["fwhm"] = 1.2

    rng = np.random.RandomState(seed=116)
    md = metadetect.Metadetect(
        config, mbobs, rng,
    )
    md.go()
    res = md.result

    config["weight"]["fwhm_reg"] = 0.8
    rng = np.random.RandomState(seed=116)
    md = metadetect.Metadetect(
        config, mbobs, rng,
    )
    md.go()
    res_reg = md.result

    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        msk = res[shear][model + "_flags"] == 0
        msk_reg = res_reg[shear][model + "_reg0.80_" + "flags"] == 0
        assert np.allclose(
            res_reg[shear][model + "_reg0.80" + "_T"][msk_reg],
            res[shear][model + "_T"][msk]
        )
        assert not np.allclose(
            res_reg[shear][model + "_reg0.80" + "_g"][msk_reg],
            res[shear][model + "_g"][msk]
        )


def test_metadetect_fitter_multi_meas():
    pytest.importorskip("sxdes")

    nband = 3
    rng = np.random.RandomState(seed=116)

    sim = Sim(rng, config={"nband": nband})
    mbobs = sim.get_mbobs()

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    del config["model"]
    del config["weight"]
    config["fitters"] = [
        {"model": "wmom", "weight": {"fwhm": 1.2}},
        {"model": "pgauss", "weight": {"fwhm": 2.0}},
        {"model": "pgauss", "weight": {"fwhm": 2.0, "fwhm_reg": 0.8}},
        {"model": "am"},
        {"model": "gauss"},
    ]

    rng = np.random.RandomState(seed=116)
    md = metadetect.Metadetect(
        config, mbobs, rng,
    )
    md.go()
    res = md.result

    model = "pgauss"
    for shear in ["noshear", "1p", "1m", "2p", "2m"]:
        msk = res[shear][model + "_flags"] == 0
        msk_reg = res[shear][model + "_reg0.80_" + "flags"] == 0
        msk_wmom = res[shear]["wmom_flags"] == 0
        msk_admom = res[shear]["am_flags"] == 0
        msk_gauss = res[shear]["gauss_flags"] == 0
        assert np.allclose(
            res[shear][model + "_reg0.80" + "_T"][msk_reg],
            res[shear][model + "_T"][msk]
        )
        assert not np.allclose(
            res[shear][model + "_reg0.80" + "_g"][msk_reg],
            res[shear][model + "_g"][msk]
        )
        assert not np.allclose(
            res[shear]["wmom_T"][msk_wmom],
            res[shear][model + "_T"][msk]
        )
        assert not np.allclose(
            res[shear]["wmom_g"][msk_wmom],
            res[shear][model + "_g"][msk]
        )

        # admom can fail so look at intersection
        assert not np.allclose(
            res[shear]["am_T"][msk_admom & msk],
            res[shear][model + "_T"][msk_admom & msk]
        )
        assert not np.allclose(
            res[shear]["am_g"][msk_admom & msk],
            res[shear][model + "_g"][msk_admom & msk]
        )

        # gauss can fail so look at intersection
        assert not np.allclose(
            res[shear]["gauss_T"][msk_gauss & msk],
            res[shear][model + "_T"][msk_gauss & msk]
        )
        assert not np.allclose(
            res[shear]["gauss_g"][msk_gauss & msk],
            res[shear][model + "_g"][msk_gauss & msk]
        )


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
@pytest.mark.parametrize("nband,nshear", [(3, 2), (1, 1), (4, 2), (3, 1)])
def test_metadetect_flux(model, nband, nshear):
    """
    test full metadetection w/ fluxes
    """
    pytest.importorskip("sxdes")

    ntrial = 1
    rng = np.random.RandomState(seed=116)

    tm0 = time.time()

    sim = Sim(rng, config={"nband": nband})
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for shear_bands in itertools.combinations(list(range(nband)), nshear):
            res = metadetect.do_metadetect(
                config, mbobs, rng, shear_band_combs=[shear_bands],
                det_band_combs="shear_bands",
            )
            for shear in ["noshear", "1p", "1m", "2p", "2m"]:
                assert np.all(res[shear]["mfrac"] == 0)
                assert np.all(
                    res[shear]["shear_bands"] == "".join("%s" % b for b in shear_bands)
                )
                assert np.all(
                    res[shear]["det_bands"] == "".join("%s" % b for b in shear_bands)
                )
                for c in res[shear].dtype.names:
                    if c.endswith("band_flux"):
                        if nband > 1:
                            assert res[shear][c][0].shape == (nband,)
                        else:
                            assert res[shear][c][0].shape == tuple()

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("coadd", [True, False])
@pytest.mark.parametrize("det_bands", [None, "shear_bands", "single"])
@pytest.mark.parametrize("model", ["wmom", "pgauss", "am", "gauss"])
def test_metadetect_multiband(model, det_bands, coadd):
    """
    test full metadetection w/ multiple bands
    """
    pytest.importorskip("sxdes")

    nband = 3
    ntrial = 1
    rng = np.random.RandomState(seed=116)

    tm0 = time.time()

    sim = Sim(rng, config={"nband": nband})
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model
    config["coadd"] = coadd

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        shear_band_combs = [list(range(nband))]
        shear_band_combs += [
            list(shear_bands)
            for shear_bands in itertools.combinations(list(range(nband)), 2)
        ]
        shear_band_combs += [
            list(shear_bands)
            for shear_bands in itertools.combinations(list(range(nband)), 1)
        ]
        det_band_combs = (
            det_bands
            if det_bands != "single"
            else [[0]] * len(shear_band_combs)
        )
        res = metadetect.do_metadetect(
            config, mbobs, rng, shear_band_combs=shear_band_combs,
            det_band_combs=det_band_combs,
        )
        if det_band_combs is None:
            det_band_combs = [list(range(nband))] * len(shear_band_combs)
        elif det_band_combs == "shear_bands":
            det_band_combs = shear_band_combs

        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.all(res[shear]["mfrac"] == 0)
            for det_bands, shear_bands in zip(det_band_combs, shear_band_combs):
                assert np.any(
                    (
                        res[shear]["shear_bands"]
                        == "".join("%s" % b for b in shear_bands)
                    )
                    & (
                        res[shear]["det_bands"]
                        == "".join("%s" % b for b in det_bands)
                    )
                )
                for c in res[shear].dtype.names:
                    if c.endswith("band_flux"):
                        if nband > 1:
                            assert res[shear][c][0].shape == (nband,)
                        else:
                            assert res[shear][c][0].shape == tuple()

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


def test_metadetect_with_color_is_same():
    pytest.importorskip("sxdes")

    model = "wmom"
    nband = 3
    ntrial = 1

    tm0 = time.time()

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = model

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        shear_band_combs = [list(range(nband))]
        shear_band_combs += [
            list(shear_bands)
            for shear_bands in itertools.combinations(list(range(nband)), 2)
        ]
        shear_band_combs += [
            list(shear_bands)
            for shear_bands in itertools.combinations(list(range(nband)), 1)
        ]

        rng = np.random.RandomState(seed=116)
        sim = Sim(rng, config={"nband": nband})
        mbobs = sim.get_mbobs()
        rng = np.random.RandomState(seed=11)
        res = metadetect.do_metadetect(
            config, mbobs, rng, shear_band_combs=shear_band_combs,
        )

        rng = np.random.RandomState(seed=116)
        sim = Sim(rng, config={"nband": nband})
        mbobs = sim.get_mbobs()
        rng = np.random.RandomState(seed=11)
        res_color = metadetect.do_metadetect(
            config, mbobs, rng, shear_band_combs=shear_band_combs,
            color_key_func=lambda x: "blah", color_dep_mbobs={"blah": mbobs},
        )
        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            for col in res[shear].dtype.names:
                assert col in res_color[shear].dtype.names
                if col == "shear_bands" or col == "det_bands":
                    assert np.array_equal(
                        res[shear][col],
                        res_color[shear][col],
                    )
                else:
                    np.testing.assert_allclose(
                        res[shear][col],
                        res_color[shear][col],
                        atol=0,
                        rtol=0,
                        equal_nan=True,
                    )

            for shear_bands in shear_band_combs:
                assert np.any(
                    res[shear]["shear_bands"] == "".join("%s" % b for b in shear_bands)
                )
                assert np.any(
                    res_color[shear]["shear_bands"]
                    == "".join("%s" % b for b in shear_bands)
                )

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("mask_region", [1, 7])
def test_fill_in_mask_col(mask_region):
    rng = np.random.RandomState(seed=10)

    rows = np.array([31.2])
    cols = np.array([51.7])
    mask = rng.randint(low=0, high=64, size=(100, 100))

    vals = metadetect._fill_in_mask_col(
        mask_region=mask_region,
        rows=rows,
        cols=cols,
        mask=mask)

    row = 31
    col = 52
    if mask_region == 1:
        assert vals[0] == mask[row, col]
    else:
        assert vals[0] == np.bitwise_or.reduce(
            mask[
                row-mask_region:row+mask_region+1,
                col-mask_region:col+mask_region+1
            ]
        )[0]


def test_get_psf_stats():
    rng = np.random.RandomState(seed=10)
    sim = Sim(rng)
    mbobs = sim.get_mbobs()
    fitting.fit_all_psfs(mbobs, rng)

    psf_stats = metadetect._get_psf_stats(mbobs, 0)
    assert psf_stats["flags"] == 0
    assert np.isfinite(psf_stats["g1"])
    assert np.isfinite(psf_stats["g2"])
    assert np.isfinite(psf_stats["T"])

    psf_stats = metadetect._get_psf_stats(mbobs, 2)
    assert psf_stats["flags"] == (procflags.PSF_FAILURE | 2)
    assert not np.isfinite(psf_stats["g1"])
    assert not np.isfinite(psf_stats["g2"])
    assert not np.isfinite(psf_stats["T"])

    for obslist in mbobs:
        for obs in obslist:
            obs.weight = -1.0*obs.weight
    psf_stats = metadetect._get_psf_stats(mbobs, 0)
    assert psf_stats["flags"] == procflags.PSF_FAILURE
    assert not np.isfinite(psf_stats["g1"])
    assert not np.isfinite(psf_stats["g2"])
    assert not np.isfinite(psf_stats["T"])

    e1s = np.arange(len(mbobs)) + 0.1
    e2s = 2*np.arange(len(mbobs)) + 0.1
    Ts = 3*np.arange(len(mbobs)) + 0.1
    wgts = np.arange(len(mbobs)) + 1
    for i, obslist in enumerate(mbobs):
        for obs in obslist:
            obs.weight = 0*obs.weight + wgts[i]
            obs.psf.meta["result"]["e"] = (e1s[i], e2s[i])
            obs.psf.meta["result"]["T"] = Ts[i]

    psf_stats = metadetect._get_psf_stats(mbobs, 0)
    assert psf_stats["flags"] == 0
    assert np.allclose(psf_stats["g1"], np.sum(wgts * e1s)/np.sum(wgts))
    assert np.allclose(psf_stats["g2"], np.sum(wgts * e2s)/np.sum(wgts))
    assert np.allclose(psf_stats["T"], np.sum(wgts * Ts)/np.sum(wgts))
