import numpy as np
import galsim
import galsim.des
import matplotlib.pyplot as plt
import seaborn as sns

from fit_des_psf import ShpPSF

sns.set()

psf = ShpPSF('shp_psf.fit')
pex = None
# psf = galsim.des.DES_PSFEx('psfcat.psf', wcs=galsim.PixelScale(0.263))
# pex = galsim.des.DES_PSFEx('psfcat.psf', wcs=galsim.PixelScale(0.263))

y, x = np.mgrid[0:1000:50, 0:1000:50]
x = x.ravel()
y = y.ravel()

sigma = []
fwhm = []
g1 = []
g2 = []

for _x, _y in zip(x, y):
    img = psf.getPSF(galsim.PositionD(x=_x, y=_y)).drawImage(
        nx=33,
        ny=33,
        scale=0.263)
    admom = galsim.hsm.FindAdaptiveMom(img)

    if pex is not None:
        pex_img = pex.getPSF(galsim.PositionD(x=_x, y=_y)).drawImage(
            nx=33,
            ny=33)
        pex_admom = galsim.hsm.FindAdaptiveMom(pex_img)
        g1.append(admom.observed_shape.g1 - pex_admom.observed_shape.g1)
        g2.append(admom.observed_shape.g2 - pex_admom.observed_shape.g2)
        sigma.append(admom.moments_sigma - pex_admom.moments_sigma)
        fwhm.append(img.calculateFWHM() - pex_img.calculateFWHM())

        plt.figure()
        ax = plt.gca()
        ax.grid(False)
        sns.heatmap(np.arcsinh(img.array - pex_img.array), ax=ax)
        plt.show()

        import pdb
        pdb.set_trace()
    else:
        g1.append(admom.observed_shape.g1)
        g2.append(admom.observed_shape.g2)
        sigma.append(admom.moments_sigma)
        fwhm.append(img.calculateFWHM())

sigma = np.array(sigma)
g1 = np.array(g1)
g2 = np.array(g2)
fwhm = np.array(fwhm)

tcks = np.arange(20) * 50 + 25

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

ax = axs[0, 0]
sns.heatmap(
    fwhm.reshape(20, 20),
    xticklabels=tcks,
    yticklabels=tcks,
    square=True,
    ax=ax)
ax.set_title('FWHM [arcsec]')

ax = axs[0, 1]
sns.heatmap(
    sigma.reshape(20, 20),
    xticklabels=tcks,
    yticklabels=tcks,
    square=True,
    ax=ax)
ax.set_title('ADMOM sigma')


ax = axs[1, 0]
sns.heatmap(
    g1.reshape(20, 20),
    xticklabels=tcks,
    yticklabels=tcks,
    square=True,
    ax=ax)
ax.set_title('g1')

ax = axs[1, 1]
sns.heatmap(
    g2.reshape(20, 20),
    xticklabels=tcks,
    yticklabels=tcks,
    square=True,
    ax=ax)
ax.set_title('g2')

plt.show()
