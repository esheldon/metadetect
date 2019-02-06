import numba
import numpy as np
import pytest
import time

from psf_homogenizer import convolve_variable_kernel


def kernel(row, col, size):
    rng = np.random.RandomState(row * 1000 + col)
    kern = np.exp(rng.normal(size=(size, size)))
    kern /= np.sum(kern)
    return kern


@pytest.mark.parametrize('im_size_i', [1, 2, 3, 10, 35, 44])
@pytest.mark.parametrize('im_size_j', [1, 2, 3, 10, 35, 44])
@pytest.mark.parametrize('kern_size', [1, 3, 5, 7, 11, 17])
def test_convolve_compare_ref(im_size_i, im_size_j, kern_size):
    rng = np.random.RandomState(seed=10)
    img = rng.uniform(size=(im_size_i, im_size_j))

    def _kern(row, col):
        return kernel(row, col, kern_size)

    imgo = convolve_variable_kernel(_kern, img)
    imgo_ref = simple_convolve_variable_kernel(_kern, img)
    assert np.allclose(imgo, imgo_ref)


def test_convolve_compare_time():
    rng = np.random.RandomState(seed=10)
    img = rng.uniform(size=(200, 200))

    def _kern(row, col):
        return kernel(row, col, 7)

    convolve_variable_kernel(_kern, img)
    t0 = time.time()
    convolve_variable_kernel(_kern, img)
    t0 = time.time() - t0

    simple_convolve_variable_kernel(_kern, img)
    t0_ref = time.time()
    simple_convolve_variable_kernel(_kern, img)
    t0_ref = time.time() - t0_ref
    assert t0 < t0_ref


@numba.jit
def simple_convolve_variable_kernel(kernel, image):
    """Convolve a variable kernel into an image.

    NOTE: This is a reference implementation used to help in testing.

    Parameters
    ----------
    kernel : callable
        A function that takes in the location in the image
        in the form of a zero-indexed (row, col) and outputs the
        kernel for that location in the image. The function signature should
        be `kernel(row, col)`.
    image : array-like, two-dimensional
        The image with which to convolve the kernel.

    Returns
    -------
    conv : np.ndarray, two-dimensional
        The convolved image.
    """
    im_new = np.zeros_like(image)
    i_shape = image.shape[0]
    j_shape = image.shape[1]
    nk = kernel(0, 0).shape[0]
    dk = (nk-1) // 2
    for i_new in range(i_shape):
        for j_new in range(j_shape):
            k = kernel(i_new, j_new)

            i_old_start = max(0, i_new-dk)
            j_old_start = max(0, j_new-dk)

            i_old_end = min(i_new + dk + 1, i_shape)
            j_old_end = min(j_new + dk + 1, j_shape)

            i_kern_start = max(0, dk-i_new)
            j_kern_start = max(0, dk-j_new)

            n_i_kern = i_old_end - i_old_start
            n_j_kern = j_old_end - j_old_start

            _sum = 0.0
            for _i in range(n_i_kern):
                for _j in range(n_j_kern):
                    _sum += (
                        image[i_old_start + _i, j_old_start + _j] *
                        k[i_kern_start + _i, j_kern_start + _j])
            im_new[i_new, j_new] = _sum
    return im_new
