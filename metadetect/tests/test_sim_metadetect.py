"""
test using lsst simple sim
"""
import numpy as np
import ngmix

CONFIG = {
    'bmask_flags': 0,
    'metacal': {
        'use_noise_image': True,
        'psf': 'fitgauss',
    },
    'psf': {
        'model': 'gauss',
        'lm_pars': {},
        'ntry': 2,
    },
    'weight': {
        'fwhm': 1.2,
    },
    'meds': {},
}


def test_sim_metadetect_smoke(ntrial=1, show=False):
    try:
        from descwl_shear_sims import Sim
        from descwl_coadd import CoaddObsSimple
        from ..sim_metadetect import SimMetadetect

        rng = np.random.RandomState()

        sim = Sim(
            rng=rng,
            epochs_per_band=3,
        )
        data = sim.gen_sim()

        # faking ngmix MultiBandObsList
        # note data is an OrderedDict
        coadd_mbobs = ngmix.MultiBandObsList()
        for band in data:
            coadd_obs = CoaddObsSimple(data[band])
            obslist = ngmix.ObsList()
            obslist.append(coadd_obs)
            coadd_mbobs.append(obslist)

        md = SimMetadetect(CONFIG, coadd_mbobs, rng)
        md.go()
        res = md.result  # noqa

    except ImportError as err:
        print(str(err))
