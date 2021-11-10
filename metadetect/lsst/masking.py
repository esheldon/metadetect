EXPAND_RAD = 16
AP_RAD = 1.5


def apply_apodized_masks(mbobs, masks, wcs):
    """
    Apply bright object masks with apodization and expansion.  The .image,
    .noise and .bmask for each observation are modified in place

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The data to mask.  The .bmask are modified in place
    masks: structured array
        Array with fields ra, dec, radius_pixels
    wcs: A stack wcs object
        For converting ra, dec to x, y
    """
    import lsst.afw.image as afw_image
    from ..masking import apply_foreground_masking_corrections

    afw_image.Mask.addMaskPlane('BRIGHT')
    afw_image.Mask.addMaskPlane('BRIGHT_EXPANDED')
    bright = afw_image.Mask.getPlaneBitMask('BRIGHT')
    bright_expanded = afw_image.Mask.getPlaneBitMask('BRIGHT_EXPANDED')

    xm, ym = wcs.skyToPixelArray(
        ra=masks['ra'], dec=masks['dec'], degrees=True,
    )
    rm = masks['radius_pixels']

    apply_foreground_masking_corrections(
        mbobs=mbobs,
        xm=xm,
        ym=ym,
        rm=rm,
        method="apodize",
        mask_expand_rad=EXPAND_RAD,
        ap_rad=AP_RAD,
        mask_bit_val=bright,
        expand_mask_bit_val=bright_expanded,
        symmetrize=False,

        # below are unused for apodized masks
        interp_bit_val=None,
        iso_buff=None,
        rng=None,
    )
