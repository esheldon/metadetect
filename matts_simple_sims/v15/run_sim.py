import sys
import pickle

import numpy as np
import joblib

from test_sim_utils import Sim, TEST_METADETECT_CONFIG
from metadetect.metadetect_and_cal import MetadetectAndCal
from metadetect.metadetect import Metadetect

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()
    HAVE_MPI = True
except Exception:
    n_ranks = 1
    rank = 0
    comm = None

    HAVE_MPI = False


DO_COMM = False
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
offset = rank * n_sims

sims = [joblib.delayed(_func)(i + offset) for i in range(n_sims)]
outputs = joblib.Parallel(
    verbose=20,
    n_jobs=-1,
    pre_dispatch='2*n_jobs',
    max_nbytes=None)(sims)

pres, mres = zip(*outputs)

pres, mres = _cut(pres, mres)

if comm is not None and DO_COMM:
    if rank == 0:
        n_recv = 0
        while n_recv < n_ranks - 1:
            status = MPI.Status()
            data = comm.recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status)
            n_recv += 1
            pres.extend(data[0])
            mres.extend(data[1])
    else:
        comm.send((pres, mres), dest=0, tag=0)
else:
    with open('data%d.pkl' % rank, 'wb') as fp:
        pickle.dump((pres, mres), fp)

if HAVE_MPI:
    comm.Barrier()

if rank == 0:
    if not DO_COMM:
        for i in range(1, n_ranks):
            with open('data%d.pkl' % i, 'rb') as fp:
                data = pickle.load(fp)
                pres.extend(data[0])
                mres.extend(data[1])

    mn, msd = _fit_m(pres, mres)

    print("""\
# of sims: {n_sims}
run: {kind}
m: {mn:f} +/- {msd:f}""".format(
        n_sims=len(pres),
        kind=kind,
        mn=mn,
        msd=msd), flush=True)

    mn, msd = _fit_m_single(pres)

    print("""\
# of sims: {n_sims}
run: {kind}
m: {mn:f} +/- {msd:f}""".format(
        n_sims=len(pres),
        kind=kind,
        mn=mn,
        msd=msd), flush=True)
