import numpy as np

EXPAND_RAD = 16
AP_RAD = 1.5


def apply_apodized_masks_mbexp(
    mbexp, bright_info, noise_mbexp=None, mfrac_mbexp=None, ormasks=None,
):
    """
    Apply bright object masks with apodization and expansion.  The .image,
    .noise and .bmask for each observation are modified in place

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The data to mask.  The image and mask are modified, with the mask
        value set to BRIGHT or BRIGHT_EXPANDED for the expanded mask.
    bright_info: structured array
        Array with fields ra, dec, radius_pixels
    noise_mbexp: lsst.afw.image.MultibandExposure
        Optional noise data to mask.  The image and mask are modified.
    mfrac_mbexp: lsst.afw.image.MultibandExposure
        Optional mfrac data to mask.  The image and mask are modified, with
        mfrac set to 1 everywhere that apodization is applied
    ormasks: list of arrays, optional
        A list of masks to logically or with the bright mask
    """
    import lsst.afw.image as afw_image
    from ..masking import (
        make_foreground_apodization_mask, make_foreground_bmask,
    )
    bands = mbexp.filters

    afw_image.Mask.addMaskPlane('BRIGHT')
    afw_image.Mask.addMaskPlane('BRIGHT_EXPANDED')
    bright = afw_image.Mask.getPlaneBitMask('BRIGHT')
    bright_expanded = afw_image.Mask.getPlaneBitMask('BRIGHT_EXPANDED')

    wcs = mbexp[bands[0]].getWcs()
    xm, ym = wcs.skyToPixelArray(
        ra=bright_info['ra'], dec=bright_info['dec'], degrees=True,
    )
    rm = bright_info['radius_pixels']

    dims = mbexp[bands[0]].image.array.shape

    ap_mask = make_foreground_apodization_mask(
        xm=xm,
        ym=ym,
        rm=rm,
        dims=dims,
        symmetrize=False,
        ap_rad=AP_RAD,
    )

    msk = np.where(ap_mask < 1)
    if msk[0].size > 0:
        for band in bands:

            exps = [mbexp[band]]
            if noise_mbexp is not None:
                exps.append(noise_mbexp[band])

            for exp in exps:
                exp.image.array[msk] *= ap_mask[msk]
                exp.variance.array[msk] = np.inf
                exp.mask.array[msk] |= bright

            if ormasks is not None:
                for ormask in ormasks:
                    ormask[msk] |= bright

            if mfrac_mbexp is not None:
                mfrac_mbexp[band].image.array[msk] = 1.0

    expanded_bmask = make_foreground_bmask(
        xm=xm,
        ym=ym,
        rm=rm + EXPAND_RAD,
        dims=dims,
        symmetrize=False,
        mask_bit_val=bright_expanded,
    )
    if np.any(expanded_bmask):
        for band in bands:
            exps = [mbexp[band]]
            if noise_mbexp is not None:
                exps.append(noise_mbexp[band])

            for exp in exps:
                exp.mask.array[:, :] |= expanded_bmask

            if ormasks is not None:
                for ormask in ormasks:
                    ormask |= expanded_bmask
