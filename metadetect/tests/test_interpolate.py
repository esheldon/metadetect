import numpy as np

from ..interpolate import (
    interpolate_image_and_noise,
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


def test_interpolate_image_and_noise_weight():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 0
    weight[30:35, 40:45] = 0.0

    # put nans here to make sure interp is done ok
    msk = weight <= 0
    image[msk] = np.nan

    rng = np.random.RandomState(seed=42)
    noises = [
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
    ]
    iimage, inoises = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noises=noises)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noises = [
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
    ]
    for noise, inoise in zip(noises, inoises):
        assert not np.allclose(noise[msk], inoise[msk])
        assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_image_and_noise_weight_fill():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 0
    weight[30:50, 40:60] = 0.0

    # put nans here to make sure interp is done ok
    msk = weight <= 0
    image[msk] = np.nan

    rng = np.random.RandomState(seed=42)
    noises = [
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
    ]
    rng = np.random.RandomState(seed=42)
    iimage, inoises = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noises=noises,
        rng=rng,
        fill_isolated_with_noise=True,
    )

    assert not np.allclose(iimage, 10 + x*5)
    assert np.allclose(np.mean(iimage[35:45, 45:55]), 0, rtol=0, atol=0.2)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noises = [
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
    ]
    for noise, inoise in zip(noises, inoises):
        assert not np.allclose(noise[msk], inoise[msk])
        assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_image_and_noise_bmask():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 1

    rng = np.random.RandomState(seed=42)
    bmask[30:35, 40:45] = 1
    bmask[:, 0] = 2
    bmask[:, -1] = 4

    # put nans here to make sure interp is done ok
    msk = (bmask & bad_flags) != 0
    image[msk] = np.nan

    rng = np.random.RandomState(seed=42)
    noises = [
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
    ]
    iimage, inoises = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noises=noises)

    assert np.allclose(iimage, 10 + x*5)

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noises = [
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
    ]
    for noise, inoise in zip(noises, inoises):
        assert not np.allclose(noise[msk], inoise[msk])
        assert np.allclose(noise[~msk], inoise[~msk])


def test_interpolate_image_and_noise_big_missing():
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 1

    rng = np.random.RandomState(seed=42)
    nse = rng.normal(size=image.shape)
    bmask[15:80, 15:80] = 1

    # put nans here to make sure interp is done ok
    msk = (bmask & bad_flags) != 0
    image[msk] = np.nan

    iimage, inoises = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noises=[nse])

    # interp will be waaay off but shpuld have happened
    assert np.all(np.isfinite(iimage))

    # make sure noise field was inteprolated
    rng = np.random.RandomState(seed=42)
    noise = rng.normal(size=image.shape)
    assert not np.allclose(noise[msk], inoises[0][msk])
    assert np.allclose(noise[~msk], inoises[0][~msk])


def test_interpolate_image_and_noise_allbad():
    # linear image interp should be perfect for regions smaller than the
    # patches used for interpolation
    y, x = np.mgrid[0:100, 0:100]
    image = (10 + x*5).astype(np.float32)
    weight = np.ones_like(image)
    bmask = np.zeros_like(image, dtype=np.int32)
    bad_flags = 1

    rng = np.random.RandomState(seed=42)
    bmask[:, :] = 1

    # put nans here to make sure interp is done ok
    msk = (bmask & bad_flags) != 0
    image[msk] = np.nan

    rng = np.random.RandomState(seed=42)
    noises = [
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
        rng.normal(size=image.shape),
    ]
    iimage, inoises = interpolate_image_and_noise(
        image=image,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noises=noises)

    assert iimage is None
    assert inoises is None


def test_interpolate_gauss_image(show=False):
    """
    test that our interpolation works decently for a linear
    piece missing from a gaussian image
    """

    rng = np.random.RandomState(seed=31415)
    noise = 0.001

    sigma = 4.0
    is2 = 1.0/sigma**2
    dims = 51, 51
    cen = (np.array(dims)-1.0)/2.0

    rows, cols = np.mgrid[
        0:dims[0],
        0:dims[1],
    ]
    rows = rows - cen[0]
    cols = cols - cen[1]

    image_unmasked = np.exp(-0.5*(rows**2 + cols**2)*is2)
    weight = image_unmasked*0 + 1.0/noise**2

    noise_image = rng.normal(scale=noise, size=image_unmasked.shape)

    badcol = int(cen[1]-3)
    bw = 3
    rr = badcol-bw, badcol+bw+1

    weight[rr[0]:rr[1], badcol] = 0.0
    image_masked = image_unmasked.copy()
    image_masked[rr[0]:rr[1], badcol] = 0.0

    bmask = np.zeros_like(image_unmasked, dtype=np.int32)
    bad_flags = 0

    iimage, inoises = interpolate_image_and_noise(
        image=image_masked,
        weight=weight,
        bmask=bmask,
        bad_flags=bad_flags,
        noises=[noise_image],
    )

    maxdiff = np.abs(image_unmasked-iimage).max()

    if show:
        import images
        images.view_mosaic([image_masked, weight])

        images.compare_images(
            image_unmasked,
            iimage,
            width=2000,
            height=int(2000*2/3),
        )
        print('max diff:', maxdiff)

    assert maxdiff < 0.0025
