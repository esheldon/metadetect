import numpy as np

from ..interpolate import (
    interpolate_image_at_mask,
)


def test_interpolate_image_at_mask():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    bmask = np.zeros_like(image, dtype=bool)
    bmask[30:35, 40:45] = True

    # put nans here to make sure interp is done ok
    image[bmask] = np.nan

    iimage = interpolate_image_at_mask(
        image=image,
        bad_msk=bmask,
    )
    assert np.allclose(iimage, 10 + x*5)


def test_interpolate_image_at_mask_allbad():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    bmask = np.zeros_like(image, dtype=bool)
    bmask[:, :] = True

    iimage = interpolate_image_at_mask(
        image=image,
        bad_msk=bmask,
    )
    assert iimage is None
