import numpy as np
from numba import njit


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
        The radius for the star mask
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
def _build_mask_image(rows, cols, radius_pixels, mask):
    nrows, ncols = mask.shape
    nmasked = 0

    for istar in range(rows.size):
        row = rows[istar]
        col = cols[istar]

        rad = radius_pixels[istar]
        rad2 = rad * rad

        if not _intersects(row, col, rad, nrows, ncols):
            continue

        for irow in range(nrows):
            rowdiff2 = (row - irow)**2
            for icol in range(ncols):

                r2 = rowdiff2 + (col - icol)**2
                if r2 < rad2:
                    mask[irow, icol] = 1
                    nmasked += 1

    return nmasked


def build_mask_image(rows, cols, radius_pixels, dims, symmetrize=False):
    """Build image of 0 or 1 for masks.

    Parameters
    ----------
    rows, cols: arrays
        Arrays of rows/cols of the mask hole locations in the "local" coordimnates
        of the image.These positions may be off the image.
    radius_pixels: array
        The radius for each mask hole.
    dims: tuple of ints
        The shape of the mask.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.

    Returns
    -------
    mask: array
        The array to set to 1 if something is masked, 0 otherwise.
    """

    # must be native byte order for numba
    cols = cols.astype('f8')
    rows = rows.astype('f8')
    radius_pixels = radius_pixels.astype('f8')

    mask = np.zeros(dims, dtype='i4')

    _build_mask_image(rows, cols, radius_pixels, mask)

    if symmetrize:
        if mask.shape[0] != mask.shape[1]:
            raise ValueError("Only square images can be symmetrized!")
        mask |= np.rot90(mask)

    return mask


def apply_mask_mbobs(mbobs, mask_catalog, maskflags, symmetrize=False):
    """Expands masks in an mbobs. This will expand the masks by setting the
    weight to zero, mfrac to 1, and setting maskflags in the bmask for every obs.

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to modify.
    mask_catalog: np.ndarray, optional
        If not None, this array should have columns 'x', 'y' and 'radius_pixels'
        that gives the location and radius of any masked objects in the image.
        Default of None indicates no catalog.
    maskflags: int
        The bit to set in the bit mask.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.
    """
    dims = None
    for obslist in mbobs:
        for obs in obslist:
            dims = obs.image.shape
            break
    if dims is None:
        raise RuntimeError("Cannot expand the masks on an empty observation!")

    mask = build_mask_image(
        mask_catalog['y'],
        mask_catalog['x'],
        mask_catalog['radius_pixels'],
        dims,
        symmetrize=symmetrize,
    )
    msk = mask != 0
    if np.any(msk):
        for obslist in mbobs:
            for obs in obslist:
                with obs.writeable():
                    if hasattr(obs, "mfrac"):
                        obs.mfrac[msk] = 1
                    obs.weight[msk] = 0
                    obs.bmask[msk] |= maskflags


def apply_mask_mfrac(mfrac, mask_catalog, symmetrize=False):
    """Expand masks in an mfrac image. This will set mfrac to 1 in the expanded
    mask region.

    Parameters
    ----------
    mfrac: array
        The masked fraction image to modify.
    mask_catalog: np.ndarray, optional
        If not None, this array should have columns 'x', 'y' and 'radius_pixels'
        that gives the location and radius of any masked objects in the image.
        Default of None indicates no catalog.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.
    """
    mask = build_mask_image(
        mask_catalog['y'],
        mask_catalog['x'],
        mask_catalog['radius_pixels'],
        mfrac.shape,
        symmetrize=symmetrize,
    )
    msk = mask != 0
    if np.any(msk):
        mfrac[msk] = 1


def apply_mask_bit_mask(bmask, mask_catalog, maskflags, symmetrize=False):
    """Expand masks in a bit mask. This will set maskflags in the expanded
    mask region.

    Parameters
    ----------
    bmask: array
        The git mask to modify.
    mask_catalog: np.ndarray, optional
        If not None, this array should have columns 'x', 'y' and 'radius_pixels'
        that gives the location and radius of any masked objects in the image.
        Default of None indicates no catalog.
    maskflags: int
        The bit to set in the bit mask.
    symmetrize: bool, optional
        If True, will symmetrize the mask with a 90 degree rotation. Default is False.
    """
    mask = build_mask_image(
        mask_catalog['y'],
        mask_catalog['x'],
        mask_catalog['radius_pixels'],
        bmask.shape,
        symmetrize=symmetrize,
    )
    msk = mask != 0
    if np.any(msk):
        bmask[msk] |= maskflags
