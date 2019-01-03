import os
import numpy as np
import galsim
import fitsio

import pytest

from fit_des_psf import ShpPSF


def _get_terms_three(_x, _y, mnx, rx, mny, ry):
    x = (_x - mnx) / rx
    y = (_y - mny) / ry
    return np.vstack(
        [np.ones_like(x),
         x, y, x*y, x*x, y*y, x*y*y, x*x*y, x*x*x, y*y*y]).T


@pytest.fixture
def data(tmpdir_factory):
    tmpdir = tmpdir_factory.getbasetemp()

    seed = 12
    order = 3
    sigma = 0.45
    mnx = 0
    rx = 1000
    mny = 0
    ry = 1000

    n_coeffs = galsim.Shapelet(sigma=sigma, order=order).bvec.shape[0]

    y, x = np.mgrid[0:1000:100, 0:1000:100]
    x = x.ravel()
    y = y.ravel()
    pterms = _get_terms_three(x, y, mnx, rx, mny, ry)
    n_polyterms = pterms.shape[1]

    rng = np.random.RandomState(seed=seed)
    poly_coeffs = rng.normal(size=(n_polyterms, n_coeffs)) * 0.1

    # make sure things are not toooo cray
    poly_coeffs[:, 0] = 0
    poly_coeffs[0, 0] = 0.7
    poly_coeffs[1, 0] = 0.1
    poly_coeffs[2, 0] = 0.2
    poly_coeffs[:, :] = 0
    poly_coeffs[0, 0] = 0.7

    order_str = \
        "(np.ones_like(x), x, y, x*y, x*x, y*y, x*y*y, x*x*y, x*x*x, y*y*y)"
    data = np.zeros(1, dtype=[
        ('order', 'i8'),
        ('sigma', 'f8'),
        ('mx', 'f8'),
        ('rx', 'f8'),
        ('my', 'f8'),
        ('ry', 'f8'),
        ('ostr', 'S%d' % len(order_str)),
        ('coeffs', 'f8', poly_coeffs.shape)])
    data['order'] = order
    data['sigma'] = sigma
    data['mx'] = mnx
    data['rx'] = rx
    data['my'] = mny
    data['ry'] = ry
    data['ostr'] = order_str
    data['coeffs'] = poly_coeffs

    fname = os.path.join(tmpdir, 'shp_psf.fit')
    fitsio.write(fname, data, clobber=True)

    return {
        'fname': fname,
        'data': data
    }


def test_get_psf(data):
    psf = ShpPSF(fname=data['fname'])
    x = 10
    y = 15
    pos = galsim.PositionD(x=x+1, y=y+1)

    shp = psf.getPSF(pos)

    # compute bvec by hand
    _x = np.atleast_1d(x)
    _y = np.atleast_1d(y)
    pterms = _get_terms_three(
        _x, _y,
        data['data']['mx'][0],
        data['data']['rx'][0],
        data['data']['my'][0],
        data['data']['ry'][0])
    bvec = np.dot(pterms, data['data']['coeffs'][0])[0, :]
    assert np.allclose(bvec, shp.bvec)


def test_sum_psf_parts(data):
    psf = ShpPSF(fname=data['fname'])
    x = 10
    y = 15
    wcs = galsim.PixelScale(0.263)
    pos = galsim.PositionD(x=x+1, y=y+1)
    shp = psf.getPSF(pos)

    # render once versus rendering a sum
    im_once = shp.drawImage(nx=33, ny=33, wcs=wcs).array

    bvec = shp.bvec.copy()
    im_sum = np.zeros_like(im_once)
    for i, bval in enumerate(bvec):
        _bvec = np.zeros_like(bvec)
        _bvec[i] = bval
        im_sum += galsim.Shapelet(
            data['data']['sigma'][0], data['data']['order'][0], bvec=_bvec
        ).drawImage(
            nx=33,
            ny=33,
            wcs=wcs).array

    assert np.allclose(im_once, im_sum)


def test_conv_sum_psf_parts(data):
    psf = ShpPSF(fname=data['fname'])
    x = 10
    y = 15
    wcs = galsim.PixelScale(0.263)
    pos = galsim.PositionD(x=x+1, y=y+1)
    shp = psf.getPSF(pos)

    gal = galsim.Gaussian(sigma=0.7).shear(g1=0.3, g2=0.8)

    # conv and render once versus conv and rendering in a sum
    im_once = galsim.Convolve(gal, shp).drawImage(nx=33, ny=33, wcs=wcs).array

    bvec = shp.bvec.copy()
    im_sum = np.zeros_like(im_once)
    for i, bval in enumerate(bvec):
        _bvec = np.zeros_like(bvec)
        _bvec[i] = bval
        _shp = galsim.Shapelet(
            data['data']['sigma'][0], data['data']['order'][0], bvec=_bvec)
        galsim.Convolve(gal, shp)
        im_sum += galsim.Convolve(gal, _shp).drawImage(
            nx=33,
            ny=33,
            wcs=wcs).array

    assert np.allclose(im_once, im_sum)
