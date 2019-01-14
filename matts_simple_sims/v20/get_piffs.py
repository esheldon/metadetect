import os
import glob
import fitsio
import numpy as np

try:
    os.makedirs('piffs')
except Exception:
    pass

bl = np.genfromtxt(
    '/astro/u/mjarvis/work/y3_piff/y3a1-v29/psf_y3a1-v29_blacklist.txt',
    dtype=int)
bl_set = set(t for t in zip(bl[:, 0], bl[:, 1]))

d = fitsio.read('/astro/u/mjarvis/work/y3_piff/y3a1-v29/psf_y3a1-v29_riz.fits')
d_set = set(t for t in zip(d['exp'], d['ccd']))

# get PSFs in the riz ones but not in the blacklist
good_ones = list(d_set - bl_set)

seed = 190
rng = np.random.RandomState(seed=seed)

inds = rng.choice(len(good_ones), size=1000, replace=False)

for i in inds:
    expnum, ccd = good_ones[i]
    psf = (
        "/astro/u/mjarvis/work/y3_piff/y3a1-v29/{exp:d}/"
        "D{exp:08d}_*_c{ccd:02d}_*_piff.fits").format(
            exp=expnum,
            ccd=ccd
        )
    fname = glob.glob(psf)[0]

    os.system('cp %s piffs/.' % fname)
