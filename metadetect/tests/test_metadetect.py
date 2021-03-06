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
import galsim
from .. import detect
from .. import metadetect
from .. import procflags
from ..fitting import Moments

DEFAULT_SIM_CONFIG = {
    'nobj': 4,
    'nband': 3,
    'noises': (0.0005, 0.001, 0.0015, 0.002),
    'scale': 0.263,
    'psf_fwhm': 0.9,
    'dims': (225, 225),
    'flux_low': 0.5,
    'flux_high': 1.5,
    'r50_low': 0.1,
    'r50_high': 2.0,
    'g_std': 0.2,
    'fracdev_low': 0.001,
    'fracdev_high': 0.99,
    'bulge_colors': np.array([0.5, 1.0, 1.5, 2.5]),
    'disk_colors': np.array([1.25, 1.0, 0.75, 0.5]),
}

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


class Sim(dict):
    def __init__(self, rng, config=None):
        self.update(DEFAULT_SIM_CONFIG)

        if config is not None:
            self.update(config)

        self['pos_width'] = self['dims'][0]/2.0*0.5 * self['scale']
        self.rng = rng

        self._set_wcs()
        self._make_psf()
        self._gpdf = ngmix.priors.GPriorBA(
            self['g_std'],
            rng=self.rng,
        )

    def get_mbobs(self):
        """
        get a simulated MultiBandObsList
        """
        all_band_obj = self._get_band_objects()

        mbobs = ngmix.MultiBandObsList()
        for band in range(self['nband']):
            band_objects = [o[band] for o in all_band_obj]
            obj = galsim.Sum(band_objects)

            im = obj.drawImage(
                nx=self['dims'][1],
                ny=self['dims'][0],
                scale=self['scale']
            ).array

            im += self.rng.normal(scale=self['noises'][band], size=im.shape)
            wt = im*0 + 1.0/self['noises'][band]**2
            bmask = np.zeros(im.shape, dtype='i4')
            ormask = np.zeros(im.shape, dtype='i4')

            obs = ngmix.Observation(
                im,
                weight=wt,
                bmask=bmask,
                ormask=ormask,
                jacobian=self._jacobian,
                psf=self._psf_obs.copy(),
            )

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs

    def _get_r50(self):
        return self.rng.uniform(
            low=self['r50_low'],
            high=self['r50_high'],
        )

    def _get_flux(self):
        return self.rng.uniform(
            low=self['flux_low'],
            high=self['flux_high'],
        )

    def _get_fracdev(self):
        return self.rng.uniform(
            low=self['fracdev_low'],
            high=self['fracdev_high'],
        )

    def _get_g(self):
        return self._gpdf.sample2d()

    def _get_dxdy(self):
        return self.rng.uniform(
            low=-self['pos_width'],
            high=self['pos_width'],
            size=2,
        )

    def _get_band_objects(self):

        all_band_obj = []
        for i in range(self['nobj']):
            r50 = self._get_r50()
            flux = self._get_flux()
            fracdev = self._get_fracdev()
            dx, dy = self._get_dxdy()

            g1d, g2d = self._get_g()
            g1b = 0.5*g1d
            g2b = 0.5*g2d

            flux_bulge = fracdev*flux
            flux_disk = (1-fracdev)*flux

            bulge_obj = galsim.DeVaucouleurs(
                half_light_radius=r50
            ).shear(g1=g1b, g2=g2b)

            disk_obj = galsim.Exponential(
                half_light_radius=r50
            ).shear(g1=g1d, g2=g2d)

            band_objs = []
            for band in range(self['nband']):
                band_disk = \
                    disk_obj.withFlux(flux_disk*self['disk_colors'][band])
                band_bulge = \
                    bulge_obj.withFlux(flux_bulge*self['bulge_colors'][band])

                obj = galsim.Sum(band_disk, band_bulge).shift(dx=dx, dy=dy)
                obj = galsim.Convolve(obj, self._psf)
                band_objs.append(obj)

            all_band_obj.append(band_objs)

        return all_band_obj

    def _set_wcs(self):
        self._jacobian = ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=self['scale'],
        )

    def _make_psf(self):

        self._psf = galsim.Gaussian(fwhm=self['psf_fwhm'])

        psf_im = self._psf.drawImage(scale=self['scale']).array
        noise = psf_im.max()/1000.0
        weight = psf_im + 1.0/noise**2
        psf_im += self.rng.normal(
            scale=noise,
            size=psf_im.shape
        )

        cen = (np.array(psf_im.shape)-1.0)/2.0
        j = self._jacobian.copy()
        j.set_cen(row=cen[0], col=cen[1])

        psf_obs = ngmix.Observation(
            psf_im,
            weight=weight,
            jacobian=j
        )
        self._psf_obs = psf_obs


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


def test_metadetect_masking():
    """
    test full metadetection with masking
    """

    ntrial = 1
    rng = np.random.RandomState(seed=116)

    sim = Sim(rng)
    config = {}
    config.update(copy.deepcopy(TEST_METADETECT_CONFIG))
    config["model"] = "wmom"

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()

        for obslist in mbobs:
            for obs in obslist:
                obs.bmask = obs.bmask | config['maskflags']

        res = metadetect.do_metadetect(config, mbobs, rng)
        for shear in ["noshear", "1p", "1m", "2p", "2m"]:
            assert res[shear] is None


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


def make_mbobs_sim(seed, nobj, nband):
    rng = np.random.RandomState(seed=seed)

    dim = 35
    cen = (dim-1)/2
    band_mbobs_list = [[] for _ in range(nband)]
    for _ in range(nobj):
        gal = galsim.Exponential(
            half_light_radius=rng.uniform(low=0.5, high=0.7),
        ).shear(
            g1=rng.uniform(low=-0.1, high=0.1),
            g2=rng.uniform(low=-0.1, high=0.1),
        ).withFlux(
            400
        )

        for band in range(nband):
            psf = galsim.Gaussian(
                fwhm=rng.uniform(low=0.8, high=0.9),
            ).shear(
                g1=rng.uniform(low=-0.1, high=0.1),
                g2=rng.uniform(low=-0.1, high=0.1),
            )

            gs_wcs = galsim.ShearWCS(
                0.25,
                galsim.Shear(
                    g1=rng.uniform(low=-0.1, high=0.1),
                    g2=rng.uniform(low=-0.1, high=0.1),
                )
            ).jacobian()
            offset = rng.uniform(low=-0.5, high=0.5, size=2)

            obj = galsim.Convolve([gal, psf])

            im = obj.drawImage(nx=dim, ny=dim, wcs=gs_wcs, offset=offset).array
            nse = np.sqrt(np.sum(im**2)) / rng.uniform(low=10, high=100)
            im += rng.normal(size=im.shape, scale=nse)

            psf_im = psf.drawImage(nx=dim, ny=dim, wcs=gs_wcs).array
            psf_obs = ngmix.Observation(
                image=psf_im,
                jacobian=ngmix.Jacobian(
                    row=cen,
                    col=cen,
                    wcs=gs_wcs,
                )
            )
            band_mbobs_list[band].append(
                ngmix.observation.get_mb_obs(
                    ngmix.Observation(
                        image=im,
                        weight=np.ones_like(im) / nse**2,
                        jacobian=ngmix.Jacobian(
                            row=cen+offset[1],
                            col=cen+offset[0],
                            wcs=gs_wcs,
                        ),
                        bmask=np.zeros_like(im, dtype=np.int32),
                        psf=psf_obs,
                        meta={"wgt": 1.0/nse**2},
                    )
                )
            )

    return band_mbobs_list


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
