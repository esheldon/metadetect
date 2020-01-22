#!/usr/bin/env python
"""
    - do a straight coadd and store it in an ngmix Observation for the
    simple_sim, which makes perfectly registered images with a pixel scale and
    same psf and wcs for all images

    - run detection and deblending
    - run a stub for measurement on deblended images (not doing anything yet)
    - optionally make a plot and overplot detections on the image
"""
import os
import sys
import logging
import numpy as np
# import matplotlib.pyplot as plt
import ngmix

from descwl_shear_sims import Sim
from descwl_coadd.coadd import MultiBandCoadds
from metadetect.lsst_metadetect import LSSTMetadetect
import fitsio
import esutil as eu
import argparse


def make_comb_data(res):
    add_dt = [('shear_type', 'S7')]

    dlist = []
    for stype in res.keys():
        data = res[stype]
        newdata = eu.numpy_util.add_fields(data, add_dt)
        newdata['shear_type'] = stype
        dlist.append(newdata)

    return eu.numpy_util.combine_arrlist(dlist)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--ntrial', type=int, default=1)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--noise', type=float, default=180)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--cosmic-rays', action='store_true')
    parser.add_argument('--bad-columns', action='store_true')
    parser.add_argument('--show', action='store_true')

    return parser.parse_args()


def main():

    args = get_args()

    rng = np.random.RandomState(args.seed)
    config = {
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

    wcs_kws = {
        'position_angle_range': (0, 360),
        'scale_frac_std': 0.01,
        'dither_range': (-0.5, 0.5),
        'shear_std': 0.01,
    }

    logging.basicConfig(stream=sys.stdout)
    logging.getLogger('descwl_shear_testing').setLevel(
        getattr(logging, 'INFO')
    )

    dlist_p = []
    dlist_m = []

    for trial in range(args.ntrial):
        print('-'*70)
        print('trial: %d/%d' % (trial+1, args.ntrial))

        trial_seed = rng.randint(0, 2**30)

        for shear_type in ('1p', '1m'):
            print(shear_type)
            trial_rng = np.random.RandomState(trial_seed)

            if shear_type == '1p':
                shear_g1 = 0.02
            else:
                shear_g1 = -0.02

            sim = Sim(
                g1=shear_g1,
                rng=trial_rng,
                epochs_per_band=args.nepochs,
                noise_per_band=args.noise,
                wcs_kws=wcs_kws,
                coadd_dim=350,
                buff=50,
                cosmic_rays=args.cosmic_rays,
                bad_columns=args.bad_columns,
            )
            data = sim.gen_sim()

            coadd_dims = (sim.coadd_dim, )*2
            mbc = MultiBandCoadds(
                data=data,
                coadd_wcs=sim.coadd_wcs,
                coadd_dims=coadd_dims,
                byband=False,
            )

            coadd_mbobs = ngmix.MultiBandObsList(
                meta={'psf_fwhm': sim.psf_kws['fwhm']},
            )
            obslist = ngmix.ObsList()
            obslist.append(mbc.coadds['all'])
            coadd_mbobs.append(obslist)

            md = LSSTMetadetect(config, coadd_mbobs, trial_rng)
            md.go()
            res = md.result
            # print(res.keys())

            comb_data = make_comb_data(res)

            if shear_type == '1p':
                dlist_p.append(comb_data)
            else:
                dlist_m.append(comb_data)

    data_1p = eu.numpy_util.combine_arrlist(dlist_p)
    data_1m = eu.numpy_util.combine_arrlist(dlist_m)

    print('writing:', args.output)
    with fitsio.FITS(args.output, 'rw', clobber=True) as fits:
        fits.write(data_1p, extname='1p')
        fits.write(data_1m, extname='1m')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    main()
