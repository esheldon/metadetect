import numpy as np
import descwl_shear_sims
import metadetect.lsst.masking
from metadetect.lsst.masking import EXPAND_RAD
from metadetect.lsst.util import exp2obs
import metadetect.lsst.vis as vis
import ngmix


def make_lsst_sim(rng, dim, mag=22, hlr=0.5, bands=['r', 'i', 'z']):

    coadd_dim = dim
    se_dim = dim

    galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=0,
        layout='grid',
        mag=mag,
        hlr=hlr,
    )

    psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = descwl_shear_sims.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        se_dim=se_dim,
        g1=0.00,
        g2=0.00,
        psf=psf,
        bands=bands,
    )
    return sim_data


def test_apply_apodize_masks(show=False):
    ntrial = 5
    nmasked = 2

    seed = 101
    dim = 200
    rng = np.random.RandomState(seed=seed)

    for itrial in range(ntrial):
        sim_data = make_lsst_sim(rng=rng, dim=dim)

        bands = list(sim_data['band_data'].keys())
        exp = sim_data['band_data'][bands[0]][0]
        dims = exp.image.array.shape
        dim = dims[0]

        dtype = [('ra', 'f8'), ('dec', 'f8'), ('radius_pixels', 'f8')]
        masks = np.zeros(nmasked, dtype=dtype)

        wcs = exp.getWcs()
        buff = 21
        x = rng.uniform(low=buff, high=dim-buff, size=nmasked)
        y = rng.uniform(low=buff, high=dim-buff, size=nmasked)
        radius = rng.uniform(low=10, high=20, size=nmasked)
        ra, dec = wcs.pixelToSkyArray(x=x, y=y, degrees=True)
        masks['ra'] = ra
        masks['dec'] = dec
        masks['radius_pixels'] = radius

        xp, yp = wcs.skyToPixelArray(ra=ra, dec=dec, degrees=True)
        assert np.allclose(x, xp)
        assert np.allclose(y, yp)

        mbobs = ngmix.MultiBandObsList()
        noise_images = []
        for band, exps in sim_data['band_data'].items():
            exp = exps[0]
            obs = exp2obs(exp)
            noiseval = np.sqrt(1/obs.weight[0, 0])
            noise_image = rng.normal(scale=noiseval, size=dims)
            obs.noise = noise_image
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

            noise_images.append(noise_image)

        metadetect.lsst.masking.apply_apodized_masks(
            mbobs=mbobs, masks=masks, wcs=wcs,
        )
        if show:
            vis.show_multi_mbobs(mbobs)

        bright = exp.mask.getPlaneBitMask('BRIGHT')
        bright_expanded = exp.mask.getPlaneBitMask('BRIGHT_EXPANDED')

        ygrid, xgrid = np.mgrid[0:dim, 0:dim]
        for iband, obslist in enumerate(mbobs):
            band = bands[iband]
            obs = obslist[0]
            exp = sim_data['band_data'][band][0]
            noise_image = noise_images[iband]

            for xx, yy, rr in zip(x, y, radius):
                ix = int(np.floor(xx+0.5))
                iy = int(np.floor(yy+0.5))
                assert obs.image[iy, ix] == 0
                assert obs.noise[iy, ix] == 0
                assert obs.bmask[iy, ix] == (bright | bright_expanded)

                # within mask the bit is set and data are modified
                r2 = (ygrid - yy)**2 + (xgrid - xx)**2
                w = np.where(r2 < rr**2)
                assert np.all(obs.bmask[w] & bright != 0)
                assert np.all(obs.weight[w] == 0.0)
                assert np.all(obs.image[w] != exp.image.array[w])
                assert np.all(obs.noise[w] == noise_image[w])

                # mask is expanded and bit is set, but data are not modified in
                # the expanded area
                w = np.where(r2 < (rr + EXPAND_RAD)**2)
                assert np.all(obs.bmask[w] & bright_expanded != 0)


if __name__ == '__main__':
    test_apply_apodize_masks(show=True)
