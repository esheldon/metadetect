import numpy as np
import descwl_shear_sims
from metadetect.lsst import masking
from metadetect.lsst import vis
from metadetect.lsst import util
from descwl_coadd.coadd import make_coadd
from descwl_coadd.coadd_nowarp import make_coadd_nowarp


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


def do_coadding(rng, sim_data, nowarp):

    bands = list(sim_data['band_data'].keys())

    if nowarp:
        coadd_data_list = [
            make_coadd_nowarp(
                exp=sim_data['band_data'][band][0],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                remove_poisson=False,
            )
            for band in bands
        ]
    else:
        coadd_data_list = [
            make_coadd(
                exps=sim_data['band_data'][band],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_bbox=sim_data['coadd_bbox'],
                remove_poisson=False,
            )
            for band in bands
        ]

    return util.extract_multiband_coadd_data(coadd_data_list)


def test_apply_apodize_masks(show=False):
    ntrial = 5
    nmasked = 2

    seed = 101
    dim = 200
    rng = np.random.RandomState(seed=seed)

    for itrial in range(ntrial):
        sim_data = make_lsst_sim(rng=rng, dim=dim)
        data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

        bands = data['mbexp'].filters

        dtype = [('ra', 'f8'), ('dec', 'f8'), ('radius_pixels', 'f8')]

        wcs = data['mbexp'][bands[0]].getWcs()

        buff = 21

        exp = data['mbexp'][bands[0]]
        dims = exp.image.array.shape
        dim = dims[0]

        x = rng.uniform(low=buff, high=dim-buff, size=nmasked)
        y = rng.uniform(low=buff, high=dim-buff, size=nmasked)
        radius = rng.uniform(low=10, high=20, size=nmasked)
        ra, dec = wcs.pixelToSkyArray(x=x, y=y, degrees=True)
        bright_info = np.zeros(nmasked, dtype=dtype)
        bright_info['ra'] = ra
        bright_info['dec'] = dec
        bright_info['radius_pixels'] = radius

        xp, yp = wcs.skyToPixelArray(ra=ra, dec=dec, degrees=True)
        assert np.allclose(x, xp)
        assert np.allclose(y, yp)

        noise_images = [
            nexp.image.array.copy() for nexp in data['noise_mbexp']
        ]

        # works in place
        masking.apply_apodized_masks_mbexp(
            mbexp=data['mbexp'],
            noise_mbexp=data['noise_mbexp'],
            mfrac_mbexp=data['mfrac_mbexp'],
            bright_info=bright_info,
            ormasks=data['ormasks'],
        )

        if show:
            vis.show_multi_mbexp(data['mbexp'])

        bright = exp.mask.getPlaneBitMask('BRIGHT')
        bright_expanded = exp.mask.getPlaneBitMask('BRIGHT_EXPANDED')

        ygrid, xgrid = np.mgrid[0:dim, 0:dim]
        for iband, band in enumerate(data['mbexp'].filters):
            exp_orig = sim_data['band_data'][band][0]
            exp = data['mbexp'][band]
            assert exp_orig is not exp
            nexp = data['noise_mbexp'][band]
            ormask = data['ormasks'][iband]

            noise_image = noise_images[iband]

            for xx, yy, rr in zip(x, y, radius):
                ix = int(np.floor(xx+0.5))
                iy = int(np.floor(yy+0.5))
                assert exp.image.array[iy, ix] == 0
                assert nexp.image.array[iy, ix] == 0
                assert exp.mask.array[iy, ix] == (bright | bright_expanded)
                assert nexp.mask.array[iy, ix] == (bright | bright_expanded)

                # within mask the bit is set and data are modified
                r2 = (ygrid - yy)**2 + (xgrid - xx)**2

                # shrink radius a little to deal with some roundoff errors
                w = np.where(r2 < (rr-0.1)**2)
                assert np.all(exp.mask.array[w] & bright != 0)
                assert np.all(nexp.mask.array[w] & bright != 0)
                assert np.all(ormask[w] & bright != 0)

                assert np.all(exp.variance.array[w] == np.inf)
                assert np.all(nexp.variance.array[w] == np.inf)

                assert np.all(exp.image.array[w] != exp_orig.image.array[w])
                assert np.all(nexp.image.array[w] != noise_image[w])

                # mask is expanded and bit is set, but data are not modified in
                # the expanded area
                w = np.where(r2 < (rr + masking.EXPAND_RAD)**2)
                assert np.all(exp.mask.array[w] & bright_expanded != 0)
                assert np.all(nexp.mask.array[w] & bright_expanded != 0)
                assert np.all(ormask[w] & bright_expanded != 0)

                w = np.where(
                    (exp.mask.array & bright == 0) &
                    (exp.mask.array & bright_expanded != 0)
                )
                assert np.all(exp.image.array[w] == exp_orig.image.array[w])
                assert np.all(nexp.image.array[w] == noise_image[w])
                assert np.all(exp.variance.array[w] != np.inf)
                assert np.all(nexp.variance.array[w] != np.inf)


if __name__ == '__main__':
    test_apply_apodize_masks(show=True)
