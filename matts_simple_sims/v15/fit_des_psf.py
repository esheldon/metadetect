import numpy as np
import galsim
import galsim.des
from scipy.optimize import curve_fit
import fitsio

PIXSCALE = galsim.PixelScale(0.263)
ORDER = 5


def _get_terms_three(_x, _y):
    x = (_x - mnx) / rx
    y = (_y - mny) / ry
    return np.vstack(
        [np.ones_like(x),
         x, y, x*y, x*x, y*y, x*y*y, x*x*y, x*x*x, y*y*y]).T


class ShpPSF(object):
    def __init__(self, fname, nx=None, ny=None):
        self.data = fitsio.read(fname)
        self._bval_cache = {}

        # cache once if desired
        if nx is not None and ny is not None:
            bvec = self._get_bvec(0, 0)
            for i in range(bvec.shape[0]):
                self._get_bval(i, nx, ny)

    def getPSF(self, pos):
        _x = np.atleast_1d(pos.x-1)
        _y = np.atleast_1d(pos.y-1)
        bvec = self._get_bvec(_x, _y)
        return galsim.Shapelet(
            self.data['sigma'][0], self.data['order'][0], bvec=bvec)

    def _get_bval(self, i, nx, ny):
        key = (i, nx, ny)
        if key not in self._bval_cache:
            # get polynomial terms for weights
            _y, _x = np.mgrid[:ny, :nx]
            _y = _y.ravel()
            _x = _x.ravel()
            pterms = self._get_terms_str(
                _x=_x,
                _y=_y,
                mx=self.data['mx'][0],
                rx=self.data['rx'][0],
                my=self.data['my'][0],
                ry=self.data['ry'][0],
                ostr=self.data['ostr'][0].decode('ascii').strip())
            self._bval_cache[key] = np.dot(
                pterms, self.data['coeffs'][0][:, i:i+1])[:, 0]
            self._bval_cache[key] = self._bval_cache[key].reshape(ny, nx)
        return self._bval_cache[key]

    def _get_terms_str(self, *, _x, _y, mx, rx, my, ry, ostr):
        x = (_x - mx) / rx  # noqa
        y = (_y - my) / ry  # noqa
        return np.vstack(eval(ostr)).T

    def _get_bvec(self, _x, _y):
        pterms = self._get_terms_str(
            _x=_x,
            _y=_y,
            mx=self.data['mx'][0],
            rx=self.data['rx'][0],
            my=self.data['my'][0],
            ry=self.data['ry'][0],
            ostr=self.data['ostr'][0].decode('ascii').strip())

        return np.dot(pterms, self.data['coeffs'][0])[0, :]


if __name__ == '__main__':
    ####################################
    # make images
    psf = galsim.des.DES_PSFEx('psfcat.psf', wcs=PIXSCALE)
    y, x = np.mgrid[0:1000:100, 0:1000:100]
    x = x.ravel()
    y = y.ravel()

    mnx = np.min(x)
    mny = np.min(y)
    rx = np.max(x) - np.min(x)
    ry = np.max(y) - np.min(y)

    ims = []
    sigmas = []
    for _x, _y in zip(x, y):
        psf_im = psf.getPSF(galsim.PositionD(x=_x+1, y=_y+1)).drawImage(
            nx=33,
            ny=33,
            wcs=PIXSCALE,
            method='no_pixel')
        ims.append(psf_im)
        sigmas.append(psf_im.calculateFWHM() / 2.355)

    print('sigma:', np.min(sigmas), np.max(sigmas), np.median(sigmas))

    ###########################################
    # do shapelets
    sigma = np.median(sigmas)

    shps = []
    for im in ims:
        shp = galsim.Shapelet.fit(sigma, ORDER, im)
        shps.append(shp.bvec.copy())
    shps = np.vstack(shps)

    ####################################
    # get poly fits
    # we have that shps = np.dot(poly_terms, poly_coeffs)
    poly_terms = _get_terms_three(x, y)
    n_params = shps.shape[1] * poly_terms.shape[1]
    poly_coeffs = np.zeros(n_params).reshape(
        poly_terms.shape[1], shps.shape[1])

    def _func(x, *p):
        return np.dot(p, x)

    for i in range(shps.shape[1]):
        opt, _ = curve_fit(_func, poly_terms.T, shps[:, i], poly_coeffs[:, i])
        poly_coeffs[:, i] = opt

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
    data['order'] = ORDER
    data['sigma'] = sigma
    data['mx'] = mnx
    data['rx'] = rx
    data['my'] = mny
    data['ry'] = ry
    data['ostr'] = order_str
    data['coeffs'] = poly_coeffs

    fitsio.write('shp_psf.fit', data, clobber=True)
