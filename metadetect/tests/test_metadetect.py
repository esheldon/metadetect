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
from .. import procflags
from ..fitting import Moments
from .sim import Sim, make_mbobs_sim

TEST_METADETECT_CONFIG = {
    "model": "wmom",

    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
    },

    'sx': {
        # in sky sigma
        # DETECT_THRESH
        'detect_thresh': 0.8,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        'deblend_cont': 0.00001,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        'minarea': 4,

        'filter_type': 'conv',

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        'filter_kernel': [
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],  # noqa
            [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],  # noqa
            [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],  # noqa
            [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],  # noqa
        ]
    },

    'meds': {
        'min_box_size': 32,
        'max_box_size': 256,

        'box_type': 'iso_radius',

        'rad_min': 4,
        'rad_fac': 2,
        'box_padding': 2,
    },

    # needed for PSF symmetrization
    'psf': {
        'model': 'gauss',

        'ntry': 2,

        'lm_pars': {
            'maxfev': 2000,
            'ftol': 1.0e-5,
            'xtol': 1.0e-5,
        }
    },

    # check for an edge hit
    'bmask_flags': 2**30,

    'maskflags': 2**0,
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
        titles=['image', 'seg', 'detim'],
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
            sx_config=config['sx'],
            meds_config=config['meds'],
        )

        mbm = mer.get_multiband_meds()

        nobj = mbm.size
        nobj_meas += nobj

        if show:
            _show_mbobs(mer)
            if ntrial > 1 and trial != (ntrial-1):
                if 'q' == input("hit a key: "):
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
                obs.bmask = obs.bmask | config['maskflags']

        mer = detect.MEDSifier(
            mbobs=mbobs,
            sx_config=config['sx'],
            meds_config=config['meds'],
            maskflags=config['maskflags'],
        )
        assert mer.cat.size == 0


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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


@pytest.mark.parametrize("model", ["wmom", "gauss"])
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
    config['model'] = model

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


@pytest.mark.parametrize('nobj', [1, 2, 11])
def test_metadetect_wavg_comp_single_band(nobj):
    """test that computing the weighted averages with one band gives the
    same result as the inputs.
    """
    # sim the mbobs list
    mbobs_list = make_mbobs_sim(134341, nobj, 1)[0]
    momres = Moments(
        {"weight": {"fwhm": 1.2}, "bmask_flags": 0},
        rng=np.random.RandomState(seed=12),
    ).go(mbobs_list)

    # now we make an Metadetect object
    # note we are making a sim here but not using it
    sim = Sim(np.random.RandomState(seed=329058), config={'nband': 1})
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = 'wmom'
    sim_mbobs = sim.get_mbobs()
    mdet = metadetect.Metadetect(config, sim_mbobs, np.random.RandomState(seed=14328))

    wgts = np.array([1])
    all_is_shear_band = [True]
    any_nonzero = False
    for i, mbobs in enumerate(mbobs_list):
        all_bres = [momres[i:i+1]]
        res = mdet._compute_wavg_fitter_mbobs_sep(
            wgts, all_bres, all_is_shear_band, mbobs
        )
        for col in [
            "wmom_T", "wmom_T_err", 'wmom_g', "wmom_g_cov", "wmom_s2n",
            "flags", "wmom_T_ratio", "wmom_flags", "psf_T", "psf_g",
        ]:
            if np.any(res[col] > 0):
                any_nonzero = True
            assert np.allclose(res[col], momres[col][i]), col

    assert any_nonzero


@pytest.mark.parametrize('nband', [2, 3, 4])
@pytest.mark.parametrize('nobj', [1, 2, 11])
def test_metadetect_wavg_comp(nband, nobj):
    """test that the weighted averages for shear are computed correctly."""
    # sim the mbobs list
    band_mbobs_list = make_mbobs_sim(134341, nobj, nband)
    band_momres = [
        Moments(
            {"weight": {"fwhm": 1.2}, "bmask_flags": 0},
            rng=np.random.RandomState(seed=12),
        ).go(mbobs_list)
        for mbobs_list in band_mbobs_list
    ]

    # now we make an Metadetect object
    # note we are making a sim here but not using it
    sim = Sim(np.random.RandomState(seed=329058), config={'nband': nband})
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = 'wmom'
    sim_mbobs = sim.get_mbobs()
    mdet = metadetect.Metadetect(config, sim_mbobs, np.random.RandomState(seed=14328))

    all_is_shear_band = [True] * nband
    any_nonzero = False
    for i in range(nobj):
        shear_mbobs = ngmix.MultiBandObsList()
        for band in range(nband):
            shear_mbobs.append(band_mbobs_list[band][i][0])
        all_bres = [momres[i:i+1] for momres in band_momres]
        wgts = np.array(
            [band_mbobs_list[b][i][0][0].meta["wgt"] for b in range(nband)]
        )
        wgts /= np.sum(wgts)
        res = mdet._compute_wavg_fitter_mbobs_sep(
            wgts, all_bres, all_is_shear_band,
            shear_mbobs,
        )
        # check a subset and don't go crazy
        for col in [
            "flags", "wmom_flags", "psf_T", "psf_g",
            "wmom_band_flux", "wmom_band_flux_err",
            "wmom_s2n", "wmom_g", "wmom_T",
        ]:
            if np.any(res[col] > 0):
                any_nonzero = True

            if col in ["psf_T", "psf_g"]:
                val = np.sum([
                    wgt * momres[col][i:i+1] for wgt, momres in zip(wgts, band_momres)
                ], axis=0)
            elif col in ["flags", "wmom_flags"]:
                val = 0
                for momres in band_momres:
                    val |= momres[col][i:i+1]
            elif col in ["wmom_band_flux", "wmom_band_flux_err"]:
                val = np.array([
                    momres[col.replace("band_", "")][i:i+1]
                    for momres in band_momres
                ]).T
            elif col in ["wmom_T"]:
                val = np.sum([
                    wgt * momres["wmom_raw_mom"][i:i+1, 1]
                    for wgt, momres in zip(wgts, band_momres)
                ], axis=0)
                val /= np.sum([
                    wgt * momres["wmom_raw_mom"][i:i+1, 0]
                    for wgt, momres in zip(wgts, band_momres)
                ], axis=0)
            elif col in ["wmom_s2n"]:
                val = np.sum([
                    wgt * momres["wmom_raw_mom"][i, 0]
                    for wgt, momres in zip(wgts, band_momres)
                ])
                val /= np.sqrt(np.sum([
                    wgt**2 * momres["wmom_raw_mom_cov"][i, 0, 0]
                    for wgt, momres in zip(wgts, band_momres)
                ]))
            elif col in ["wmom_g"]:
                val = np.sum([
                    wgt * momres["wmom_raw_mom"][i:i+1, 2:]
                    for wgt, momres in zip(wgts, band_momres)
                ], axis=0)
                val /= np.sum([
                    wgt * momres["wmom_raw_mom"][i:i+1, 1]
                    for wgt, momres in zip(wgts, band_momres)
                ], axis=0)
            else:
                assert False, "col %s not in elif block for test!" % col

            assert np.allclose(res[col], val), col

    assert any_nonzero


def test_metadetect_wavg_flagging():
    """test that the weighted averages for shear are computed correctly."""
    # sim the mbobs list
    nband = 2
    nobj = 4
    band_mbobs_list = make_mbobs_sim(134341, nobj, nband)
    band_momres = [
        Moments(
            {"weight": {"fwhm": 1.2}, "bmask_flags": 0},
            rng=np.random.RandomState(seed=12),
        ).go(mbobs_list)
        for mbobs_list in band_mbobs_list
    ]

    # now we make an Metadetect object
    # note we are making a sim here but not using it
    sim = Sim(np.random.RandomState(seed=329058), config={'nband': nband})
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = 'wmom'
    sim_mbobs = sim.get_mbobs()
    mdet = metadetect.Metadetect(config, sim_mbobs, np.random.RandomState(seed=14328))

    all_is_shear_band = [True] * nband
    for i in range(nobj):
        shear_mbobs = ngmix.MultiBandObsList()
        for band in range(nband):
            shear_mbobs.append(band_mbobs_list[band][i][0])
        all_bres = [momres[i:i+1] for momres in band_momres]
        wgts = np.array(
            [band_mbobs_list[b][i][0][0].meta["wgt"] for b in range(nband)]
        )
        wgts /= np.sum(wgts)

        nonshear_mbobs = None
        if i == 0:
            shear_mbobs[1] = ngmix.ObsList()
        elif i == 1:
            wgts[0] = 0.0
        elif i == 2:
            nonshear_mbobs = ngmix.MultiBandObsList()
            nonshear_mbobs.append(ngmix.ObsList())

        res = mdet._compute_wavg_fitter_mbobs_sep(
            wgts, all_bres, all_is_shear_band,
            shear_mbobs, nonshear_mbobs=nonshear_mbobs,
        )

        if i in [0, 1, 2]:
            assert (res['flags'] & procflags.OBJ_FAILURE) != 0
            assert (res['wmom_flags'] & procflags.OBJ_FAILURE) != 0
