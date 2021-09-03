"""
interpolation utils - orig. from beckermr/pizza-cutter
"""
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator
import logging

from numba import njit

logger = logging.getLogger(__name__)


@njit
def _get_nearby_good_pixels(bad_msk, nbad, buff, iso_buff):
    """
    get the set of good pixels surrounding bad pixels.

    Parameters
    ----------
    bad_msk : bool array
        2d array of mask bits. True means it is a bad
        pixel
    nbad : int
        The number of bad pixels.
    buff : int
        The size of the good pixel buffer around each bad pixel.
    iso_buff : int
        The size of the good pixel test buffer region around each bad pixel. If
        a given bad pixel doesn't have any good pixels in this region, then it is
        marked as isolated.

    Returns
    -------
    bad_ind : array-like
        The 1d indices of the pixels to interp in row*ncol + col.
    bad_iso : array-like
        An array of 1 if the bad pixel doesn't have any buffer pixels which are ok, 0
        otherwise.
    good_ind : array-like
        The 1d indices of the good pixels to use in the interp in row*ncol + col.
    """

    nrows, ncols = bad_msk.shape

    ngood = nbad*(2*buff+1)**2
    bad_ind = np.zeros(ngood, dtype=np.int64)
    bad_iso = np.zeros(ngood, dtype=np.int64)
    good_ind = np.zeros(ngood, dtype=np.int64)
    no_good = 1

    ibad = 0
    igood = 0
    for row in range(nrows):
        for col in range(ncols):
            val = bad_msk[row, col]
            if val:
                bad_ind[ibad] = row * ncols + col

                row_start = row - iso_buff
                row_end = row + iso_buff
                col_start = col - iso_buff
                col_end = col + iso_buff

                if row_start < 0:
                    row_start = 0
                if row_end > (nrows-1):
                    row_end = nrows-1
                if col_start < 0:
                    col_start = 0
                if col_end > (ncols-1):
                    col_end = ncols-1

                no_good = 1
                for rc in range(row_start, row_end+1):
                    for cc in range(col_start, col_end+1):
                        tval = bad_msk[rc, cc]
                        if not tval:
                            no_good = 0

                if no_good == 1:
                    bad_iso[ibad] = 1

                if buff != iso_buff:
                    row_start = row - buff
                    row_end = row + buff
                    col_start = col - buff
                    col_end = col + buff

                    if row_start < 0:
                        row_start = 0
                    if row_end > (nrows-1):
                        row_end = nrows-1
                    if col_start < 0:
                        col_start = 0
                    if col_end > (ncols-1):
                        col_end = ncols-1

                for rc in range(row_start, row_end+1):
                    for cc in range(col_start, col_end+1):
                        tval = bad_msk[rc, cc]
                        if not tval:

                            if igood == ngood:
                                raise RuntimeError('good_pix too small')

                            ind = rc*ncols + cc
                            good_ind[igood] = ind
                            igood += 1

                ibad += 1

    bad_ind = bad_ind[:ibad]
    bad_iso = bad_iso[:ibad]
    good_ind = good_ind[:igood]

    return bad_ind, bad_iso, good_ind


def interpolate_image_at_mask(
    *, image, bad_msk, maxfrac=0.90, buff=4,
    fill_isolated_with_noise=False, weight=None, rng=None, iso_buff=1,
):
    """
    interpolate the bad pixels in an image

    Parameters
    ----------
    image : array
        the pixel data
    bad_msk : array
        boolean array, True means it is a bad pixel
    maxfrac : float, optional
        If the fraction of bad pixels is greater than this,
        None is returned. Default is 0.90.
    buff : int, optional
        The buffer of good pixels around each bad pixel to keep for the interpolant.
    weight : float, optional
        The weight to use for generating noise when filling interiors of interpolated
        regions with noise.
    fill_isolated_with_noise : bool, optional
        Fill isolated bad pixels with noise and then interp.
    rng : np.random.RandomState, optional
        An RNG to use if we are filling isolated bad pixels with noise.
    iso_buff : int
        The size of the good pixel test buffer region around each bad pixel. If
        a given bad pixel doesn't have any good pixels in this region, then it is
        marked as isolated.

    Returns
    -------
    interp_image : array-like
        The interpolated image.
    """
    nrows, ncols = image.shape
    npix = bad_msk.size

    nbad = bad_msk.sum()
    bm_frac = nbad/npix
    if bm_frac <= maxfrac and nbad < npix:
        interp_image = image.copy()

        bad_ind, bad_iso, _good_ind = _get_nearby_good_pixels(
            bad_msk, nbad, buff, iso_buff,
        )
        good_ind = np.unique(_good_ind)
        good_yx = np.unravel_index(good_ind, bad_msk.shape)
        bad_yx = np.unravel_index(bad_ind, bad_msk.shape)

        if fill_isolated_with_noise:
            if rng is None:
                raise RuntimeError(
                    "You must pass an RNG to fill an image with noise "
                    "when interpolating!"
                )

            if weight is None:
                raise RuntimeError(
                    "You must pass a weight to fill an image with noise "
                    "when interpolating!"
                )

            msk = bad_iso == 1
            if np.any(msk):
                # mark them as ok pixels
                bad_msk = bad_msk.copy()
                bad_msk[bad_yx[0][msk], bad_yx[1][msk]] = False

                # keep the ones we have to fill
                noise_fill_yx = (bad_yx[0][msk], bad_yx[1][msk])

                # recompute the good pixels so that they inlcude the ones we
                # will noise fill
                nbad = bad_msk.sum()
                bad_ind, _, _good_ind = _get_nearby_good_pixels(
                    bad_msk, nbad, buff, iso_buff,
                )
                good_ind = np.unique(_good_ind)
                bad_yx = np.unravel_index(bad_ind, bad_msk.shape)
                good_yx = np.unravel_index(good_ind, bad_msk.shape)

                shape = noise_fill_yx[0].shape
                interp_image[noise_fill_yx[0], noise_fill_yx[1]] = rng.normal(
                    size=shape, scale=1.0/np.sqrt(weight)
                )

        good_pix = np.array(good_yx).T
        bad_pix = np.array(bad_yx).T
        good_im = interp_image[good_yx[0], good_yx[1]]
        img_interp = CloughTocher2DInterpolator(
            good_pix,
            good_im,
            fill_value=0.0,
        )
        interp_image[bad_msk] = img_interp(bad_pix)

        return interp_image

    else:
        return None
