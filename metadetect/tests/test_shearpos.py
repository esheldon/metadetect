"""
test with super simple sim.  The purpose here is not
to make sure it gets the right answer or anything, just
to test all the moving parts
"""
import numpy as np
import ngmix
import galsim
from .. import shearpos


def test_shear_pos(show=False):
    """
    test shearing and unshearing positions
    """

    step = 0.10

    dims = 100, 100
    im = np.zeros(dims)
    cen = (np.array(dims) - 1)/2
    jacobian = ngmix.Jacobian(
        row=cen[0],
        col=cen[1],
        dudrow=0.263,
        dudcol=-0.01,
        dvdrow=+0.01,
        dvdcol=0.263,
    )

    obs = ngmix.Observation(
        im,
        jacobian=jacobian,
    )

    shears = ['noshear', '1p', '1m', '2p', '2m']

    rows, cols = np.mgrid[
        :dims[0]:5,
        :dims[1]:5,
    ]
    rows = rows.ravel()
    cols = cols.ravel()

    for sstr in shears:
        srows, scols = shearpos.shear_positions(
            rows,
            cols,
            sstr,
            obs,
            step=step,
        )

        if show:
            _show_pos(rows, cols, srows, scols, title=sstr)

        crows, ccols = shearpos.unshear_positions(
            srows,
            scols,
            sstr,
            obs,
            step=step,
        )

        if show:
            _show_pos(rows, cols, crows, ccols, title=sstr+' unsheared')

        assert np.allclose(rows, crows), 'checking rows inverse works'
        assert np.allclose(cols, ccols), 'checking cols inverse works'


def test_shear_pos_image(show=False):
    """
    test shearing and unshearing positions against the galsim
    shearing of an image
    """

    step = 0.10

    dims = 100, 100
    im = np.zeros(dims)
    cen = (np.array(dims) - 1)/2
    jacobian = ngmix.Jacobian(
        row=cen[0],
        col=cen[1],
        dudrow=0.263,
        dudcol=-0.01,
        dvdrow=+0.01,
        dvdcol=0.263,
    )
    gs_wcs = jacobian.get_galsim_wcs()

    obs = ngmix.Observation(
        im,
        jacobian=jacobian,
    )

    # the single pixel with a non-zero value
    row, col = 35, 15
    with obs.writeable():
        obs.image[row, col] = 1

    gsim = galsim.Image(obs.image, wcs=gs_wcs)
    ii = galsim.InterpolatedImage(gsim, x_interpolant='lanczos15')

    shears = ['1p', '1m', '2p', '2m']
    for sstr in shears:
        gs = shearpos.get_galsim_shear(sstr, step)

        ii_sheared = ii.shear(g1=gs.g1, g2=gs.g2)
        gsim_sheared = gsim.copy()
        ii_sheared.drawImage(image=gsim_sheared)

        srow, scol = shearpos.shear_positions(
            row,
            col,
            sstr,
            obs,
            step=step,
        )

        if show:
            import images
            images.view_mosaic([obs.image, gsim_sheared.array])

        irow = np.rint(srow[0]).astype('i4')
        icol = np.rint(scol[0]).astype('i4')

        maxval = gsim_sheared.array.max()
        smaxval = gsim_sheared.array[irow, icol]

        assert smaxval == maxval, 'checking sheared position against image'


def _show_pos(rows, cols, srows, scols, **kw):
    import biggles

    plt = biggles.plot(
        rows,
        cols,
        visible=False,
        **kw
    )
    biggles.plot(
        srows,
        scols,
        color='red',
        plt=plt,
    )
