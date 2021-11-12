from numba import njit
import numpy as np
from .interpolate import interpolate_image_at_mask


def apply_apodization_corrections(*, mbobs, ap_rad, mask_bit_val):
    """Apply an apodization mask around the edge of the images in an mbobs to
    prevemnt FFT artifacts.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to mask
    mask_bit_val: int
        The bit to set in the bit mask for areas that are apodized.
    ap_rad: float
        When apodizing, the scale of the kernel. The total kernel goes from 0 to 1
        over 6*ap_rad.
    """
    ap_mask = np.ones_like(mbobs[0][0].image)
    _build_square_apodization_mask(ap_rad, ap_mask)

    msk = ap_mask < 1
    if np.any(msk):
        for obslist in mbobs:
            for obs in obslist:
                # the pixels list will be reset upon exiting
                with obs.writeable():
                    obs.image *= ap_mask
                    obs.noise *= ap_mask
                    obs.bmask[msk] |= mask_bit_val
                    if hasattr(obs, "mfrac"):
                        obs.mfrac[msk] = 1.0
                    if np.all(msk):
                        obs.ignore_zero_weight = False
                    obs.weight[msk] = 0.0


@njit
def _build_square_apodization_mask(ap_rad, ap_mask):
    ap_range = int(6*ap_rad + 0.5)

    ny, nx = ap_mask.shape
    for y in range(min(ap_range+1, ny)):
        for x in range(nx):
            ap_mask[y, x] *= _ap_kern_kern(y, ap_range, ap_rad)
            ap_mask[ny-1 - y, x] *= _ap_kern_kern(y, ap_range, ap_rad)

    for y in range(ny):
        for x in range(min(ap_range+1, nx)):
            ap_mask[y, x] *= _ap_kern_kern(x, ap_range, ap_rad)
            ap_mask[y, nx - 1 - x] *= _ap_kern_kern(x, ap_range, ap_rad)


def apply_foreground_masking_corrections(
    *, mbobs, xm, ym, rm, method, mask_expand_rad,
    mask_bit_val, expand_mask_bit_val, interp_bit_val,
    symmetrize, ap_rad, iso_buff, rng,
):
    """
    Apply corrections for masks of large foreground objects like local galaxies
    and stars.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to mask
    xm: np.ndarray
        The x/column location of the mask holes in zero-indexed pixels.
    ym: np.ndarray
        The y/row location of the mask holes in zero-indexed pixels.
    rm: np.ndarray
        The radii of the mask holes in pixels.
    method: str
        The method to use for masking corrections. It should be one of 'interp',
        'interp-noise', or 'apodize'.
    mask_bit_val: int
        The bit to set in the bit mask for areas inside the mask holes.
    expand_mask_bit_val: int
        The bit to set in the bit mask for areas inside the expanded mask holes.
    interp_bit_val: int
        The bit to set for areas in the mask that are interpolated.
    symmetrize: bool
        If True, the mask holes will be symmetrized via a 90 degree rotation.
    ap_rad: float
        When apodizing, the scale of the kernel. The total kernel goes from 0 to 1
        over 6*ap_rad.
    iso_buff: float
        When using 'interp-noise', the number of pixels away from a good pixel a
        given pixel must be to be noise interpolated.
    rng: np.random.RandomState
        An RNG to use when doing 'interp-noise'.
    """

    if method == 'interp':
        _apply_mask_interp(
            mbobs=mbobs,
            xm=xm,
            ym=ym,
            rm=rm,
            symmetrize=symmetrize,
            mask_bit_val=mask_bit_val,
            interp_bit_val=interp_bit_val,
            fill_isolated_with_noise=False,
            iso_buff=iso_buff,
            rng=rng,
        )
    elif method == 'interp-noise':
        _apply_mask_interp(
            mbobs=mbobs,
            xm=xm,
            ym=ym,
            rm=rm,
            symmetrize=symmetrize,
            mask_bit_val=mask_bit_val,
            interp_bit_val=interp_bit_val,
            fill_isolated_with_noise=True,
            iso_buff=iso_buff,
            rng=rng,
        )
    elif method == 'apodize':
        _apply_mask_apodize(
            mbobs=mbobs,
            xm=xm,
            ym=ym,
            rm=rm,
            symmetrize=symmetrize,
            ap_rad=ap_rad,
            mask_bit_val=mask_bit_val,
        )
    else:
        raise RuntimeError(
            "Can only do one of 'interp', 'interp-noise' or 'apodize' for "
            "handling foreground masks (got %s)!" % method
        )

    if mask_expand_rad > 0:
        expanded_bmask = make_foreground_bmask(
            xm=xm,
            ym=ym,
            rm=rm + mask_expand_rad,
            dims=mbobs[0][0].image.shape,
            symmetrize=symmetrize,
            mask_bit_val=expand_mask_bit_val,
        )
        for obslist in mbobs:
            for obs in obslist:
                # the pixels list will be reset upon exiting
                with obs.writeable():
                    obs.bmask |= expanded_bmask


def _apply_mask_interp(
    *,
    mbobs,
    xm,
    ym,
    rm,
    symmetrize,
    mask_bit_val,
    interp_bit_val,
    fill_isolated_with_noise,
    iso_buff,
    rng,
):

    # masking is same for all, just take the first
    fg_bmask = make_foreground_bmask(
        xm=xm,
        ym=ym,
        rm=rm,
        dims=mbobs[0][0].image.shape,
        symmetrize=symmetrize,
        mask_bit_val=mask_bit_val,
    )

    # now modify the masks, weight maps, and interpolate in all
    # bands
    bad_logic = fg_bmask != 0
    wbad = np.where(bad_logic)
    if wbad[0].size > 0:

        for obslist in mbobs:
            for obs in obslist:
                # the pixels list will be reset upon exiting
                with obs.writeable():
                    obs.bmask |= fg_bmask

                    if hasattr(obs, "mfrac"):
                        obs.mfrac[wbad] = 1.0
                    if np.all(bad_logic):
                        obs.ignore_zero_weight = False
                    obs.weight[wbad] = 0.0

                    if not np.all(bad_logic):
                        wmsk = obs.weight > 0
                        wgt = np.median(obs.weight[wmsk])
                        interp_image = interpolate_image_at_mask(
                            image=obs.image,
                            bad_msk=bad_logic,
                            maxfrac=1.0,
                            iso_buff=iso_buff,
                            fill_isolated_with_noise=fill_isolated_with_noise,
                            rng=rng,
                            weight=wgt,
                        )
                        interp_noise = interpolate_image_at_mask(
                            image=obs.noise,
                            bad_msk=bad_logic,
                            maxfrac=1.0,
                            iso_buff=iso_buff,
                            fill_isolated_with_noise=fill_isolated_with_noise,
                            rng=rng,
                            weight=wgt,
                        )
                    else:
                        interp_image = None
                        interp_noise = None

                    if interp_image is None or interp_noise is None:
                        obs.bmask |= mask_bit_val
                        if hasattr(obs, "mfrac"):
                            obs.mfrac[:, :] = 1.0
                        obs.ignore_zero_weight = False
                        obs.weight[:, :] = 0.0
                    else:
                        obs.image = interp_image
                        obs.noise = interp_noise
                        obs.bmask[wbad] |= interp_bit_val


def _apply_mask_apodize(
    *,
    mbobs,
    xm,
    ym,
    rm,
    symmetrize,
    ap_rad,
    mask_bit_val,
):
    obs0 = mbobs[0][0]
    ap_mask = make_foreground_apodization_mask(
        xm=xm,
        ym=ym,
        rm=rm,
        dims=obs0.image.shape,
        symmetrize=symmetrize,
        ap_rad=ap_rad,
    )

    msk = ap_mask < 1
    if np.any(msk):
        for obslist in mbobs:
            for obs in obslist:
                # the pixels list will be reset upon exiting
                with obs.writeable():
                    obs.image *= ap_mask
                    obs.noise *= ap_mask
                    obs.bmask[msk] |= mask_bit_val
                    if hasattr(obs, "mfrac"):
                        obs.mfrac[msk] = 1.0
                    if np.all(msk):
                        obs.ignore_zero_weight = False
                    obs.weight[msk] = 0.0


def make_foreground_bmask(
    *,
    xm,
    ym,
    rm,
    dims,
    symmetrize,
    mask_bit_val,
):
    """
    Make a bit mask marking the locations of holes at (xm,ym) with radii rm
    with bit mask_bit_val.

    Parameters
    ----------
    xm: np.ndarray
        The x/column location of the mask holes in zero-indexed pixels.
    ym: np.ndarray
        The y/row location of the mask holes in zero-indexed pixels.
    rm: np.ndarray
        The radii of the mask holes in pixels.
    dims: tuple of ints
        The dimensions of the mask.
    symmetrize: bool
        If True, the mask holes will be symmetrized via a 90 degree rotation.
    mask_bit_val: int
        The bit to set in the bit mask for areas inside the mask holes.

    Returns
    -------
    bmask: np.ndarray
        The bit mask.
    """

    # must be native byte order for numba
    bmask = np.zeros(dims, dtype='i4')
    _do_mask_foreground(
        rows=ym.astype('f8'),
        cols=xm.astype('f8'),
        radius_pixels=rm.astype('f8'),
        bmask=bmask,
        flag=mask_bit_val,
    )

    if symmetrize:
        bmask |= np.rot90(bmask)

    return bmask


def make_foreground_apodization_mask(
    *,
    xm,
    ym,
    rm,
    dims,
    symmetrize,
    ap_rad,
):
    """
    Make foreground apodization mask for mask holes at (xm,ym) with radius rm.

    Parameters
    ----------
    xm: np.ndarray
        The x/column location of the mask holes in zero-indexed pixels.
    ym: np.ndarray
        The y/row location of the mask holes in zero-indexed pixels.
    rm: np.ndarray
        The radii of the mask holes in pixels.
    dims: tuple of ints
        The dimensions of the mask.
    symmetrize: bool
        If True, the mask holes will be symmetrized via a 90 degree rotation.
    ap_rad: float
        When apodizing, the scale of the kernel. The total kernel goes from 0 to 1
        over 6*ap_rad.

    Returns
    -------
    ap_mask: np.ndarray
        The apodization mask.
    """

    # must be native byte order for numba
    ap_mask = np.ones(dims, dtype='f8')
    _do_apodization_mask(
        rows=ym.astype('f8'),
        cols=xm.astype('f8'),
        radius_pixels=rm.astype('f8'),
        ap_mask=ap_mask,
        ap_rad=ap_rad,
    )

    if symmetrize:
        ap_mask *= np.rot90(ap_mask)

    return ap_mask


@njit
def _intersects(row, col, radius_pixels, nrows, ncols):
    """
    low level routine to check if the mask intersects the image.
    For simplicty just check the bounding rectangle

    Parameters
    ----------
    row, col: float
        The row and column of the mask center
    radius_pixels: float
        The radius for the mask
    nrows, ncols: int
        Shape of the image

    Returns
    -------
    True if it intersects, otherwise False
    """

    low_row = -radius_pixels
    high_row = nrows + radius_pixels - 1
    low_col = -radius_pixels
    high_col = ncols + radius_pixels - 1

    if (
        row > low_row and row < high_row and
        col > low_col and col < high_col
    ):
        return True
    else:
        return False


@njit
def _ap_kern_kern(x, m, h):
    # cumulative triweight kernel
    y = (x - m) / h + 3
    if y < -3:
        return 0
    elif y > 3:
        return 1
    else:
        val = (
            -5 * y ** 7 / 69984
            + 7 * y ** 5 / 2592
            - 35 * y ** 3 / 864
            + 35 * y / 96
            + 1 / 2
        )
        return val


@njit
def _do_apodization_mask(*, rows, cols, radius_pixels, ap_mask, ap_rad):
    """low-level code to make the apodization mask

    Parameters
    ----------
    rows, cols: arrays
        Arrays of rows/cols of mask locations in the "local" pixel frame of the
        slice, not the overall pixels of the big coadd. These positions may be
        off the image.
    radius_pixels: array
        The radius for each mask.
    ap_mask: array
        The array to fill with the apodization fraction.
    ap_rad: float
        The scale in pixels for the apodization transition from 1 to 0.
    """
    ny, nx = ap_mask.shape
    ns = cols.shape[0]

    nmasked = 0
    for i in range(ns):
        x = cols[i]
        y = rows[i]
        rad = radius_pixels[i]
        rad2 = rad**2

        if not _intersects(y, x, rad, ny, nx):
            continue

        for _y in range(ny):
            dy2 = (_y - y)**2
            for _x in range(nx):
                dr2 = (_x - x)**2 + dy2
                if dr2 < rad2:
                    ap_mask[_y, _x] *= _ap_kern_kern(np.sqrt(dr2), rad, ap_rad)
                    nmasked += 1

    return nmasked


@njit
def _do_mask_foreground(*, rows, cols, radius_pixels, bmask, flag):
    """
    low level code to mask foreground objects

    Parameters
    ----------
    rows, cols: arrays
        Arrays of rows/cols of mask locations in the "local" pixel frame of the
        slice, not the overall pixels of the big coadd. These positions may be
        off the image.
    radius_pixels: array
        The radius for each mask.
    bmask: array
        The bmask to modify.
    flag: int
        The flag value to "or" into the bmask.
    """
    nrows, ncols = bmask.shape
    nmasked = 0

    for ifg in range(rows.size):
        row = rows[ifg]
        col = cols[ifg]

        rad = radius_pixels[ifg]
        rad2 = rad * rad

        if not _intersects(row, col, rad, nrows, ncols):
            continue

        for irow in range(nrows):
            rowdiff2 = (row - irow)**2
            for icol in range(ncols):

                r2 = rowdiff2 + (col - icol)**2
                if r2 < rad2:
                    bmask[irow, icol] |= flag
                    nmasked += 1

    return nmasked
