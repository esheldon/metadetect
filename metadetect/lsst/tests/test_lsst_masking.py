import numpy as np
import descwl_shear_sims
from metadetect.masking import get_ap_range
from metadetect.lsst import masking
from metadetect.lsst import vis
from metadetect.lsst import util
from metadetect.lsst.metadetect import run_metadetect
from descwl_coadd.coadd import make_coadd
from descwl_coadd.coadd_nowarp import make_coadd_nowarp
import lsst.geom as geom


def make_lsst_sim(
    rng, dim, mag=22, hlr=0.5, bands=['r', 'i', 'z'], layout='grid',
):

    coadd_dim = dim
    se_dim = dim

    galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=0,
        layout=layout,
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


def test_apply_apodized_bright_masks(show=False):
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
        masking.apply_apodized_bright_masks_mbexp(
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
            mfrac = data['mfrac_mbexp'][band]

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
                assert np.all(mfrac.image.array[w] == 1)

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


def extract_cell_mbexp(mbexp, cell_size, start_x, start_y):
    from metadetect.lsst.util import get_mbexp, copy_mbexp

    bbox_begin = mbexp.getBBox().getBegin()

    new_begin = geom.Point2I(
        x=bbox_begin.getX() + start_x,
        y=bbox_begin.getY() + start_y,
    )
    extent = geom.Extent2I(cell_size)
    new_bbox = geom.Box2I(
        new_begin,
        extent,
    )

    subexps = []
    for band in mbexp.filters:
        exp = mbexp[band]
        # we need to make a copy of it
        subexp = exp[new_bbox]

        assert np.all(
            exp.image.array[
                start_y:start_y+cell_size,
                start_x:start_x+cell_size
            ] == subexp.image.array[:, :]
        )

        subexps.append(subexp)

    return copy_mbexp(get_mbexp(subexps))


def test_apply_apodized_bright_masks_subexp(show=False):
    """
    test that this works for sub exposures
    """
    seed = 101
    dim = 200
    rng = np.random.RandomState(seed=seed)

    sim_data = make_lsst_sim(rng=rng, dim=dim)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    bands = data['mbexp'].filters

    dtype = [('ra', 'f8'), ('dec', 'f8'), ('radius_pixels', 'f8')]

    wcs = data['mbexp'][bands[0]].getWcs()

    exp = data['mbexp'][bands[0]]
    xy0 = exp.getXY0()

    # in bigger coordinate system
    x = xy0.x + 100
    y = xy0.y + 100
    radius = 20

    spt = wcs.pixelToSky(x, y)
    ra = spt.getRa().asDegrees()
    dec = spt.getDec().asDegrees()

    bright_info = np.zeros(1, dtype=dtype)
    bright_info['ra'] = ra
    bright_info['dec'] = dec
    bright_info['radius_pixels'] = radius

    xp, yp = wcs.skyToPixelArray(ra=ra, dec=dec, degrees=True)
    assert np.allclose(x, xp)
    assert np.allclose(y, yp)

    if show:
        vis.show_multi_mbexp(data['mbexp'])

    cell_size = 50
    start_x = x - xy0.x - 25
    start_y = y - xy0.y - 25
    sub_mbexp = extract_cell_mbexp(
        mbexp=data['mbexp'],
        cell_size=cell_size,
        start_x=start_x,
        start_y=start_y,
    )
    nsub_mbexp = extract_cell_mbexp(
        mbexp=data['noise_mbexp'],
        cell_size=cell_size,
        start_x=start_x,
        start_y=start_y,
    )
    mfsub_mbexp = extract_cell_mbexp(
        mbexp=data['mfrac_mbexp'],
        cell_size=cell_size,
        start_x=start_x,
        start_y=start_y,
    )
    ormasks = [
        ormask[
            start_y:start_y+cell_size,
            start_x:start_x+cell_size,
        ] for ormask in data['ormasks']
    ]

    masking.apply_apodized_bright_masks_mbexp(
        mbexp=sub_mbexp,
        noise_mbexp=nsub_mbexp,
        mfrac_mbexp=mfsub_mbexp,
        bright_info=bright_info,
        ormasks=ormasks,
    )

    bright = exp.mask.getPlaneBitMask('BRIGHT')
    bright_expanded = exp.mask.getPlaneBitMask('BRIGHT_EXPANDED')

    ygrid, xgrid = np.mgrid[0:cell_size, 0:cell_size]

    for iband, band in enumerate(data['mbexp'].filters):
        exp = sub_mbexp[band]
        nexp = nsub_mbexp[band]
        ormask = ormasks[iband]
        mfrac = mfsub_mbexp[band]

        txy0 = exp.getXY0()

        ix = int(np.floor(x+0.5)) - txy0.x
        iy = int(np.floor(y+0.5)) - txy0.y

        assert exp.image.array[iy, ix] == 0
        assert nexp.image.array[iy, ix] == 0
        assert exp.mask.array[iy, ix] == (bright | bright_expanded)
        assert nexp.mask.array[iy, ix] == (bright | bright_expanded)

        # within mask the bit is set and data are modified
        r2 = (ygrid - iy)**2 + (xgrid - ix)**2

        # shrink radius a little to deal with some roundoff errors
        w = np.where(r2 < (radius-0.1)**2)

        assert np.all(exp.mask.array[w] & bright != 0)
        assert np.all(nexp.mask.array[w] & bright != 0)
        assert np.all(ormask[w] & bright != 0)

        assert np.all(exp.variance.array[w] == np.inf)
        assert np.all(nexp.variance.array[w] == np.inf)
        assert np.all(mfrac.image.array[w] == 1)

        # mask is expanded and bit is set, but data are not modified in
        # the expanded area
        w = np.where(r2 < (radius + masking.EXPAND_RAD)**2)
        assert np.all(exp.mask.array[w] & bright_expanded != 0)
        assert np.all(nexp.mask.array[w] & bright_expanded != 0)
        assert np.all(ormask[w] & bright_expanded != 0)

        w = np.where(
            (exp.mask.array & bright == 0) &
            (exp.mask.array & bright_expanded != 0)
        )
        assert np.all(exp.variance.array[w] != np.inf)
        assert np.all(nexp.variance.array[w] != np.inf)


def test_apply_apodized_edge_masks(show=False):
    seed = 101
    dim = 250
    rng = np.random.RandomState(seed=seed)

    sim_data = make_lsst_sim(rng=rng, dim=dim)
    data = do_coadding(rng=rng, sim_data=sim_data, nowarp=True)

    noise_images = [
        nexp.image.array.copy() for nexp in data['noise_mbexp']
    ]

    # works in place
    masking.apply_apodized_edge_masks_mbexp(
        mbexp=data['mbexp'],
        noise_mbexp=data['noise_mbexp'],
        mfrac_mbexp=data['mfrac_mbexp'],
        ormasks=data['ormasks'],
    )

    if show:
        vis.show_multi_mbexp(data['mbexp'])

    band0 = data['mbexp'].filters[0]
    edge = data['mbexp'][band0].mask.getPlaneBitMask('APODIZED_EDGE')

    ygrid, xgrid = np.mgrid[0:dim, 0:dim]

    ap_range = get_ap_range(masking.AP_RAD)
    w = np.where(
        (xgrid < ap_range) |
        (ygrid < ap_range) |
        (xgrid > (dim - ap_range - 1)) |
        (ygrid > (dim - ap_range - 1))
    )
    for iband, band in enumerate(data['mbexp'].filters):
        exp_orig = sim_data['band_data'][band][0]
        exp = data['mbexp'][band]
        assert exp_orig is not exp
        nexp = data['noise_mbexp'][band]
        mfrac = data['mfrac_mbexp'][band]
        ormask = data['ormasks'][iband]

        noise_image = noise_images[iband]

        assert np.all(exp.mask.array[w] & edge != 0)
        assert np.all(nexp.mask.array[w] & edge != 0)
        assert np.all(ormask[w] & edge != 0)

        assert np.all(exp.variance.array[w] == np.inf)
        assert np.all(nexp.variance.array[w] == np.inf)
        assert np.all(mfrac.image.array[w] == 1)

        assert np.all(exp.image.array[w] != exp_orig.image.array[w])
        assert np.all(nexp.image.array[w] != noise_image[w])


def test_apply_apodized_bright_masks_metadetect(show=False):
    """
    look for detected objects with expanded mask set
    """
    ntrial = 5
    nmasked = 2

    seed = 101
    dim = 200
    rng = np.random.RandomState(seed=seed)

    found = False
    for itrial in range(ntrial):
        sim_data = make_lsst_sim(
            rng=rng, dim=dim, layout='random', bands=['i'],
        )

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

        # works in place
        masking.apply_apodized_bright_masks_mbexp(
            mbexp=data['mbexp'],
            noise_mbexp=data['noise_mbexp'],
            mfrac_mbexp=data['mfrac_mbexp'],
            bright_info=bright_info,
            ormasks=data['ormasks'],
        )

        res = run_metadetect(config=None, rng=rng, **data)

        if show:
            vis.show_multi_mbexp(data['mbexp'], sources=res['noshear'])

        # bright = exp.mask.getPlaneBitMask('BRIGHT')
        bright_expanded = exp.mask.getPlaneBitMask('BRIGHT_EXPANDED')

        if np.any(res['noshear']['bmask'] & bright_expanded != 0):
            found = True
            break

    assert found


if __name__ == '__main__':
    # test_apply_apodized_bright_masks(show=True)
    test_apply_apodized_bright_masks_subexp(show=True)
    # test_apply_apodized_edge_masks(show=True)
    # test_apply_apodized_bright_masks_metadetect(show=True)
