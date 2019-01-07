import os
import sys
import pickle
import tqdm

import numpy as np

from test_sim_utils import Sim, TEST_METADETECT_CONFIG
from metadetect.metadetect_and_cal import MetadetectAndCal
from metadetect.metadetect import Metadetect

DO_MDET = False


def _meas_shear(res):
    op = res['1p']
    q = (op['flags'] == 0) & (op['wmom_s2n'] > 10) & (op['wmom_T_ratio'] > 1.2)
    if not np.any(q):
        return None
    g1p = op['wmom_g'][q, 0]

    om = res['1m']
    q = (om['flags'] == 0) & (om['wmom_s2n'] > 10) & (om['wmom_T_ratio'] > 1.2)
    if not np.any(q):
        return None
    g1m = om['wmom_g'][q, 0]

    o = res['noshear']
    q = (o['flags'] == 0) & (o['wmom_s2n'] > 10) & (o['wmom_T_ratio'] > 1.2)
    if not np.any(q):
        return None
    g1 = o['wmom_g'][q, 0]

    return np.mean(g1p), np.mean(g1m), np.mean(g1)


def _cut(prr, mrr):
    prr_keep = []
    mrr_keep = []
    for pr, mr in zip(prr, mrr):
        if pr is None or mr is None:
            continue
        prr_keep.append(pr)
        mrr_keep.append(mr)
    return prr_keep, mrr_keep


def _get_stuff(rr):
    _a = np.vstack(rr)
    g1p = _a[:, 0]
    g1m = _a[:, 1]
    g1 = _a[:, 2]

    return g1, (g1p - g1m) / 2 / 0.01 * 0.02


def _fit_m(prr, mrr):
    g1p, R11p = _get_stuff(prr)
    g1m, R11m = _get_stuff(mrr)

    x = (R11p + R11m)/2
    y = (g1p - g1m)/2

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in range(10000):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


def _fit_m_single(prr):
    g1p, R11p = _get_stuff(prr)

    x = R11p
    y = g1p

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in range(10000):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


if DO_MDET:
    def _run_sim_mdet(seed):
        rng = np.random.RandomState(seed=seed)
        mbobs = Sim(rng, config={'g1': 0.02}).get_mbobs()
        md = Metadetect(config, mbobs, rng)
        md.go()
        pres = _meas_shear(md.result)

        rng = np.random.RandomState(seed=seed)
        mbobs = Sim(rng, config={'g1': -0.02}).get_mbobs()
        md = Metadetect(config, mbobs, rng)
        md.go()
        mres = _meas_shear(md.result)

        return pres, mres

    kind = 'mdet'
    _func = _run_sim_mdet
else:
    def _run_sim_mdetcal(seed):
        rng = np.random.RandomState(seed=seed)
        sim = Sim(rng, config={'g1': 0.02})
        mbobs = sim.get_mbobs()
        jac_func = sim.get_wcs_jac_func()
        psf_rec_funcs = sim.get_psf_rec_funcs()
        md = MetadetectAndCal(
            config, mbobs, rng,
            wcs_jacobian_func=jac_func,
            psf_rec_funcs=psf_rec_funcs)
        md.go()
        pres = _meas_shear(md.result)

        rng = np.random.RandomState(seed=seed)
        sim = Sim(rng, config={'g1': -0.02})
        mbobs = sim.get_mbobs()
        jac_func = sim.get_wcs_jac_func()
        psf_rec_funcs = sim.get_psf_rec_funcs()
        md = MetadetectAndCal(
            config, mbobs, rng,
            wcs_jacobian_func=jac_func,
            psf_rec_funcs=psf_rec_funcs)
        md.go()
        mres = _meas_shear(md.result)

        return pres, mres

    kind = 'mdetcal'
    _func = _run_sim_mdetcal

print('running measurement: %s' % kind, flush=True)

config = {}
config.update(TEST_METADETECT_CONFIG)

n_sims = int(sys.argv[1])
seed = int(sys.argv[2])
odir = sys.argv[3]

seeds = np.random.RandomState(seed).randint(
    low=0,
    high=2**30,
    size=n_sims)

with tqdm.tqdm(seeds) as sitr:
    outputs = [_func(seed) for seed in sitr]

pres, mres = zip(*outputs)
pres, mres = _cut(pres, mres)

with open(os.path.join(odir, 'data_%d.pkl' % seed), 'wb') as fp:
    pickle.dump((pres, mres), fp)
