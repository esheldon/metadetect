import time
import copy
import numpy as np
import ngmix
import galsim
import metadetect
import tqdm
import joblib

import pytest


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
        'max_box_size': 32,

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


def make_sim(
    *,
    seed,
    g1,
    g2,
    dim=251,
    buff=34,
    scale=0.25,
    dens=100,
    ngrid=7,
    snr=1e6,
):
    rng = np.random.RandomState(seed=seed)

    half_loc = (dim-buff*2)*scale/2

    if ngrid is None:
        area_arcmin2 = ((dim - buff*2)*scale/60)**2
        nobj = int(dens * area_arcmin2)
        x = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
        y = rng.uniform(low=-half_loc, high=half_loc, size=nobj)
    else:
        half_ngrid = (ngrid-1)/2
        x, y = np.meshgrid(np.arange(ngrid), np.arange(ngrid))
        x = (x.ravel() - half_ngrid)/half_ngrid * half_loc
        y = (y.ravel() - half_ngrid)/half_ngrid * half_loc
        nobj = x.shape[0]

    cen = (dim-1)/2
    psf_dim = 53
    psf_cen = (psf_dim-1)/2

    psf = galsim.Gaussian(fwhm=0.9)
    gals = []
    for ind in range(nobj):
        u, v = rng.uniform(low=-scale, high=scale, size=2)
        u += x[ind]
        v += y[ind]
        gals.append(galsim.Exponential(half_light_radius=0.5).shift(u, v))
    gals = galsim.Add(gals)
    gals = gals.shear(g1=g1, g2=g2)
    gals = galsim.Convolve([gals, psf])

    im = gals.drawImage(nx=dim, ny=dim, scale=scale).array
    psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, scale=scale).array

    nse = (
        np.sqrt(np.sum(
            galsim.Convolve([
                psf,
                galsim.Exponential(half_light_radius=0.5),
            ]).drawImage(scale=0.25).array**2)
        )
        / snr
    )

    im += rng.normal(size=im.shape, scale=nse)
    wgt = np.ones_like(im) / nse**2
    jac = ngmix.DiagonalJacobian(scale=scale, row=cen, col=cen)
    psf_jac = ngmix.DiagonalJacobian(scale=scale, row=psf_cen, col=psf_cen)

    obs = ngmix.Observation(
        image=im,
        weight=wgt,
        jacobian=jac,
        ormask=np.zeros_like(im, dtype=np.int32),
        bmask=np.zeros_like(im, dtype=np.int32),
        psf=ngmix.Observation(
            image=psf_im,
            jacobian=psf_jac,
        ),
    )
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)
    return mbobs


def _shear_cuts(arr, model):
    if model == "wmom":
        tmin = 1.2
    else:
        tmin = 0.5
    msk = (
        (arr['flags'] == 0)
        & (arr[f'{model}_s2n'] > 10)
        & (arr[f'{model}_T_ratio'] > tmin)
    )
    return msk


def _meas_shear_data(res, model):
    msk = _shear_cuts(res['noshear'], model)
    g1 = np.mean(res['noshear'][f'{model}_g'][msk, 0])
    g2 = np.mean(res['noshear'][f'{model}_g'][msk, 1])

    msk = _shear_cuts(res['1p'], model)
    g1_1p = np.mean(res['1p'][f'{model}_g'][msk, 0])
    msk = _shear_cuts(res['1m'], model)
    g1_1m = np.mean(res['1m'][f'{model}_g'][msk, 0])
    R11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res['2p'], model)
    g2_2p = np.mean(res['2p'][f'{model}_g'][msk, 1])
    msk = _shear_cuts(res['2m'], model)
    g2_2m = np.mean(res['2m'][f'{model}_g'][msk, 1])
    R22 = (g2_2p - g2_2m) / 0.02

    dt = [
        ('g1', 'f8'),
        ('g2', 'f8'),
        ('R11', 'f8'),
        ('R22', 'f8')]
    return np.array([(g1, g2, R11, R22)], dtype=dt)


def _bootstrap_stat(d1, d2, func, seed, nboot=500):
    dim = d1.shape[0]
    rng = np.random.RandomState(seed=seed)
    stats = []
    for _ in tqdm.trange(nboot, leave=False):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind], d2[ind]))
    return stats


def meas_m_c_cancel(pres, mres):
    x = np.mean(pres['g1'] - mres['g1'])/2
    y = np.mean(pres['R11'] + mres['R11'])/2
    m = x/y/0.02 - 1

    x = np.mean(pres['g2'] + mres['g2'])/2
    y = np.mean(pres['R22'] + mres['R22'])/2
    c = x/y

    return m, c


def boostrap_m_c(pres, mres):
    m, c = meas_m_c_cancel(pres, mres)
    bdata = _bootstrap_stat(pres, mres, meas_m_c_cancel, 14324, nboot=500)
    merr, cerr = np.std(bdata, axis=0)
    return m, merr, c, cerr


def run_sim(seed, mdet_seed, model, **kwargs):
    mbobs_p = make_sim(seed=seed, g1=0.02, g2=0.0, **kwargs)
    cfg = copy.deepcopy(TEST_METADETECT_CONFIG)
    cfg["model"] = model
    _pres = metadetect.do_metadetect(
        copy.deepcopy(cfg),
        mbobs_p,
        np.random.RandomState(seed=mdet_seed)
    )
    if _pres is None:
        return None

    mbobs_m = make_sim(seed=seed, g1=-0.02, g2=0.0, **kwargs)
    _mres = metadetect.do_metadetect(
        copy.deepcopy(cfg),
        mbobs_m,
        np.random.RandomState(seed=mdet_seed)
    )
    if _mres is None:
        return None

    return _meas_shear_data(_pres, model), _meas_shear_data(_mres, model)


@pytest.mark.parametrize(
    'model,snr,ngrid,ntrial', [
        ("wmom", 1e6, 7, 50),
        ("ksigma", 1e6, 7, 50),
        ("wmom", 1e6, None, 10000),
    ]
)
def test_shear_meas(model, snr, ngrid, ntrial):
    nsub = max(ntrial // 100, 50)
    nitr = ntrial // nsub
    rng = np.random.RandomState(seed=116)
    seeds = rng.randint(low=1, high=2**29, size=ntrial)
    mdet_seeds = rng.randint(low=1, high=2**29, size=ntrial)

    tm0 = time.time()

    print("")

    pres = []
    mres = []
    loc = 0
    for itr in tqdm.trange(nitr):
        jobs = [
            joblib.delayed(run_sim)(
                seeds[loc+i], mdet_seeds[loc+i], model, snr=snr, ngrid=ngrid,
            )
            for i in range(nsub)
        ]
        outputs = joblib.Parallel(n_jobs=-1, verbose=100, backend='loky')(jobs)

        for out in outputs:
            if out is None:
                continue
            pres.append(out[0])
            mres.append(out[1])
        loc += nsub

        m, merr, c, cerr = boostrap_m_c(
            np.concatenate(pres),
            np.concatenate(mres),
        )
        print(
            (
                "\n"
                "nsims: %d\n"
                "m [1e-3, 3sigma]: %s +/- %s\n"
                "c [1e-5, 3sigma]: %s +/- %s\n"
                "\n"
            ) % (
                len(pres),
                m/1e-3,
                3*merr/1e-3,
                c/1e-5,
                3*cerr/1e-5,
            ),
            flush=True,
        )

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial, flush=True)

    pres = np.concatenate(pres)
    mres = np.concatenate(mres)
    m, merr, c, cerr = boostrap_m_c(pres, mres)

    print(
        (
            "\n\nm [1e-3, 3sigma]: %s +/- %s"
            "\nc [1e-5, 3sigma]: %s +/- %s"
        ) % (
            m/1e-3,
            3*merr/1e-3,
            c/1e-5,
            3*cerr/1e-5,
        ),
        flush=True,
    )

    assert np.abs(m) < max(1e-3, 3*merr)
    assert np.abs(c) < 3*cerr
