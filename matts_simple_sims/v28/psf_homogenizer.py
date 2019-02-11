import numpy as np

import galsim


class PSFHomogenizer(object):
    """Homogenize the PSF using GalSim.

    This class homogenizes the PSF using GalSim to perform the
    necessary image manipulations.

    Parameters
    ----------
    psf_model : function
        A function with call signature `psf_model(row, col)` that takes the
        zero-indexed position in the image and outputs an `np.ndarray` with the
        PSF image. This array should be square and have an odd size.
    image_shape : list-like, length 2
        The shape of the image to homogenize. This shape is used to search
        for the target PSF for homogenization. The output image will have
        approximately this target PSF.
    patch_size : int, optional
        The size in pixels of the patches used to process the image. A
        separate homogenization kernel is built for each patch. Thus a patch
        size of 1 will build a separate kernel for each pixel in the image. A
        patch size greater than 1 will build the homogenization kernel using
        PSF at the center of the patch and use this kernel to compute the whole
        patch. The patch size must be at least 1 and it must divide the image
        exactly.
    sigma : float, optional
        The sigma of the Gaussian kernel used to regularize the homogenization
        process. The input units are pixels, so that 0.1 is one tenth of a
        pixel.

    Methods
    -------
    homogenize_image(image)
        Homogenize the PSF of an image.
    get_kernel(row, col, sampling_factor=1)
        Get an image of the homogenization kernel.
    get_target_psf()
        Get an image of the target PSF.
    """
    def __init__(self, psf_model, image_shape, patch_size=5, sigma=1e-6):
        self.psf_model = psf_model
        self.image_shape = tuple(image_shape)
        self.patch_size = patch_size
        self.sigma = sigma

        assert patch_size > 0
        assert (image_shape[0] // patch_size) * patch_size == image_shape[0]
        assert (image_shape[1] // patch_size) * patch_size == image_shape[1]

        self._set_target_psf()

    def homogenize_image(self, image):
        """Homogenize the PSF of an image.

        Parameters
        ----------
        image : np.ndarray
            The image to which to apply the PSF homogenization.

        Returns
        -------
        himage : np.ndarray
            The PSF homogenized image.
        """
        assert self.image_shape == image.shape

        # offset to the center of each patch
        patch_size_2 = (self.patch_size - 1) / 2

        # this size is the patch size plus all of the pixels needed
        # to form at least one PSF image. thus we subtract 1 since the
        # patch_size will always be at least 1
        tot_size = self.patch_size + self._psf_im_shape[0] - 1

        # offset of central patch_size pixels in the output of the
        # homogenization convolution (of size tot_size)
        psf_size_2 = (self._psf_im_shape[0] - 1) // 2

        # output image is the same size as the input
        himage = np.zeros_like(image)

        # we pad the input with the zeros needed for both halves of the
        # psf image width
        pimage = np.pad(image, psf_size_2, 'constant')

        n_row_patches = self.image_shape[0] // self.patch_size
        n_col_patches = self.image_shape[1] // self.patch_size
        for row_patch in range(n_row_patches):
            # start of the patch in the final image and the buffered image
            row = row_patch * self.patch_size

            for col_patch in range(n_col_patches):
                col = col_patch * self.patch_size

                patch = galsim.InterpolatedImage(
                    galsim.ImageD(pimage[row:row+tot_size,
                                         col:col+tot_size].copy()),
                    wcs=galsim.PixelScale(1.0))

                psf_im = self.psf_model(
                    row + patch_size_2,
                    col + patch_size_2)
                psf_im /= np.sum(psf_im)
                gim = galsim.InterpolatedImage(
                    galsim.ImageD(psf_im),
                    wcs=galsim.PixelScale(1))

                kern = galsim.Convolve(
                    self._target_psf,
                    galsim.Deconvolve(gim))

                hpatch = galsim.Convolve(patch, kern).drawImage(
                    nx=tot_size,
                    ny=tot_size,
                    wcs=galsim.PixelScale(1),
                    method='no_pixel')

                himage[row:row+self.patch_size, col:col+self.patch_size] \
                    = hpatch.array[psf_size_2:psf_size_2+self.patch_size,
                                   psf_size_2:psf_size_2+self.patch_size]

        return himage

    def get_kernel(self, row, col, sampling_factor=1):
        """Get an image of the homogenization kernel.

        This kernel should not be used directly. This method is only
        for visualization purposes.

        Parameters
        ----------
        sampling_factor : float, optional
            A sampling factor by which to oversample the kernel image. A
            sampling factor of 2 indicates that the output image pixels
            are half the size of the original image pixels.

        Returns
        -------
        kern : np.ndarray
            An image of the homogenization kernel.
        """
        psf_im = self.psf_model(row, col)
        psf_im /= np.sum(psf_im)
        gim = galsim.InterpolatedImage(
            galsim.ImageD(psf_im),
            wcs=galsim.PixelScale(1))
        kern = galsim.Convolve(self._target_psf, galsim.Deconvolve(gim))
        return kern.drawImage(
            nx=self._psf_im_shape[1],
            ny=self._psf_im_shape[0],
            scale=1.0 / sampling_factor,
            method='no_pixel').array

    def get_target_psf(self):
        """Get an image of the target PSF.

        Returns
        -------
        psf : np.ndarray
            An image of the target PSF.
        """
        return self._target_psf.drawImage(
            nx=self._psf_im_shape[1],
            ny=self._psf_im_shape[0],
            scale=1.0,
            method='no_pixel').array

    def _set_target_psf(self):
        # find the largets kernel according to
        #  sqrt(T) * dilation
        # where
        # g = sqrt(shear.g1**2 + shear.g2**2)
        # dilation = 1.0 + 2.0*g
        self._target_psf_image = None
        self._target_psf_size = None
        self._psf_im_shape = None
        ni = self.image_shape[0] // self.patch_size
        nj = self.image_shape[1] // self.patch_size
        dg = (self.patch_size - 1) / 2

        for i in range(ni+1):
            row = min(i * self.patch_size + dg, self.image_shape[0])
            for j in range(nj+1):
                col = min(j * self.patch_size + dg, self.image_shape[1])

                psf_im = self.psf_model(row, col)
                psf_im /= np.sum(psf_im)
                if self._psf_im_shape is None:
                    self._psf_im_shape = psf_im.shape
                    assert self._psf_im_shape[0] % 2 == 1
                    assert self._psf_im_shape[1] % 2 == 1
                    assert self._psf_im_shape[0] == self._psf_im_shape[1]
                else:
                    assert self._psf_im_shape == psf_im.shape

                hsmpars = galsim.hsm.HSMParams(
                    max_mom2_iter=1000)
                gim = galsim.ImageD(psf_im, wcs=galsim.PixelScale(1))
                try:
                    moms = galsim.hsm.FindAdaptiveMom(gim, hsmparams=hsmpars)
                    fac = gim.calculateFWHM() * (
                        1.0 + 2.0 * np.sqrt(
                            moms.observed_shape.g1**2 +
                            moms.observed_shape.g2**2))
                    if (self._target_psf_size is None or
                            fac > self._target_psf_size):
                        self._target_psf_size = fac
                        self._target_psf_image = psf_im
                        self._target_psf_loc = (row, col)
                except galsim.errors.GalSimHSMError:
                    pass

        # the final PSF is an interpolated image convolved with the Gaussian
        # smoothing kernel
        self._target_psf = galsim.Convolve(
                galsim.InterpolatedImage(
                    galsim.ImageD(self._target_psf_image),
                    wcs=galsim.PixelScale(1)),
                galsim.Gaussian(sigma=self.sigma))
