"""
test with super simple sim.  The purpose here is not
to make sure it gets the right answer or anything, just
to test all the moving parts
"""
import time
import pytest
import copy
import numpy as np
import ngmix

from .. import detect
from .. import metadetect
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

    # needed for PSF symmetrization
    "psf": {
        "model": "gauss",

        "ntry": 2,

        "lm_pars": {
            "maxfev": 2000,
            "ftol": 1.0e-5,
            "xtol": 1.0e-5,
        }
    },

    # check for an edge hit
    "bmask_flags": 2**30,

    "maskflags": 2**0,
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
    rng = np.random.RandomState(seed=45)

    sim = Sim(rng)

    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        for obslist in mbobs:
            for obs in obslist:
                obs.bmask = obs.bmask | config["maskflags"]

        mer = detect.MEDSifier(
            mbobs=mbobs,
            sx_config=config["sx"],
            meds_config=config["meds"],
            maskflags=config["maskflags"],
        )
        assert mer.cat.size == 0


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
def test_metadetect(model):
    """
    test full metadetection
    """
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
            msk = res[shear]['flags'] == 0
            for col in res[shear].dtype.names:
                assert np.all(np.isfinite(res[shear][msk][col])), (
                    "result column '%s' has NaNs: %s" % (col, res[shear][msk][col]))

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
def test_metadetect_mfrac(model):
    """
    test full metadetection w/ mfrac
    """
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
            msk = res[shear]['flags'] == 0
            for col in res[shear].dtype.names:
                assert np.all(np.isfinite(res[shear][msk][col])), (
                    "result column '%s' has NaNs: %s" % (col, res[shear][msk][col]))

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
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


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
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


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
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


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
def test_metadetect_maskflags_all(model):
    """
    test full metadetection w/ all bmask all maskflags
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


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
def test_metadetect_bmask_some(model):
    """
    test full metadetection w/ some bmask all maskflags
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


@pytest.mark.parametrize("model", ["wmom", "pgauss", "ksigma"])
@pytest.mark.parametrize("nband,nshear", [(3, 2), (1, 1), (4, 2), (3, 1)])
def test_metadetect_flux(model, nband, nshear):
    """
    test full metadetection w/ fluxes
    """
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
        shear_mbobs = ngmix.MultiBandObsList()
        nonshear_mbobs = ngmix.MultiBandObsList()
        for i in range(len(mbobs)):
            if i < nshear:
                shear_mbobs.append(mbobs[i])
            else:
                nonshear_mbobs.append(mbobs[i])
        if len(nonshear_mbobs) == 0:
            nonshear_mbobs = None
        res = metadetect.do_metadetect(
            config, shear_mbobs, rng, nonshear_mbobs=nonshear_mbobs
        )
        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert np.all(res[shear]["mfrac"] == 0)
            for c in res[shear].dtype.names:
                if c.endswith("band_flux"):
                    if nband > 1:
                        assert res[shear][c][0].shape == (nband,)
                    else:
                        assert res[shear][c][0].shape == tuple()

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)
