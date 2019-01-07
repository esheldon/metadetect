import glob
import pickle
import tqdm
import numpy as np


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

    x = R11p + R11m
    y = g1p - g1m

    rng = np.random.RandomState(seed=100)
    mvals = []
    for _ in tqdm.trange(1000):
        ind = rng.choice(len(y), replace=True, size=len(y))
        mvals.append(np.mean(y[ind]) / np.mean(x[ind]) - 1)

    return np.mean(y) / np.mean(x) - 1, np.std(mvals)


files = glob.glob('outputs/data*.pkl')

pres = []
mres = []
for fname in tqdm.tqdm(files):
    with open(fname, 'rb') as fp:
        data = pickle.load(fp)
        pres.extend(data[0])
        mres.extend(data[1])

pres, mres = _cut(pres, mres)
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
