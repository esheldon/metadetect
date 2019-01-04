import sys
import os
import pickle
import tqdm
import numpy as np

def _get_stuff(rr):
    g1p = np.array([np.mean(r[0]) for r in rr])
    g1m = np.array([np.mean(r[1]) for r in rr])
    g1 = np.array([np.mean(r[2]) for r in rr])

    return g1, (g1p - g1m) / 2 / 0.01 * 0.02


def _fit_m(prr, mrr):
    g1p, R11p = _get_stuff(prr)
    g1m, R11m = _get_stuff(mrr)

    x = R11p + R11m
    y = g1p - g1m

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in tqdm.trange(10000):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


pres = []
mres = []
for i in tqdm.trange(1, 3):
    if not os.path.exists('data%d.pkl' % i):
        continue
    with open('data%d.pkl' % i, 'rb') as fp:
        data = pickle.load(fp)
        pres.extend(data[0])
        mres.extend(data[1])
        
mn, msd = _fit_m(pres, mres)

kind = 'mdetcal'

print("""\
# of sims: {n_sims}
run: {kind}
m: {mn:f} +/- {msd:f}""".format(
    n_sims=len(pres),
    kind=kind,
    mn=mn,
    msd=msd), flush=True)
