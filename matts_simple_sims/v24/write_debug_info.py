import pickle
import numpy as np
import tqdm

from test_sim_utils import Sim, TEST_METADETECT_CONFIG
from metadetect.metadetect_and_cal import MetadetectAndCal
from metadetect.metadetect import Metadetect


def _run_sim_mdet(seed):
    rng = np.random.RandomState(seed=seed)
    mbobs = Sim(rng, config={'g1': 0.02}).get_mbobs()
    md = Metadetect(config, mbobs, rng)
    md.go()
    return md.result


def _run_sim_mdetcal(seed):
    rng = np.random.RandomState(seed=seed)
    sim = Sim(rng, config={'g1': 0.02})
    mbobs = sim.get_mbobs()
    jac_func = sim.get_wcs_jac_func()
    psf_rec_funcs = sim.get_psf_rec_funcs()
    md = MetadetectAndCal(
        config, mbobs, rng,
        wcs_jacobian_func=jac_func,
        psf_rec_funcs=psf_rec_funcs,
        force_mdet_psf=True)
    md.go()
    return md.result


config = {}
config.update(TEST_METADETECT_CONFIG)


for seed in tqdm.trange(10, 100):
    res_md = _run_sim_mdet(seed)
    res_mdc = _run_sim_mdetcal(seed)

    with open('mdet%d.pkl' % seed, 'wb') as fp:
        pickle.dump(res_md, fp)

    with open('mdetcal%d.pkl' % seed, 'wb') as fp:
        pickle.dump(res_mdc, fp)
