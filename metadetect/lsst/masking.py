EXPAND_RAD = 16  # confirm with Matt
AP_RAD = 1


def apply_apodized_masks(mbobs, masks):
    """
    Apply bright object masks with apodization and expansion

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The data to mask.  The .bmask are modified in place
    masks: structured array
        Array with fields ra, dec, radius_pixels
    """
    import lsst.afw.image as afw_image
    from ..masking import apply_foreground_masking_corrections

    afw_image.Mask.addMaskPlane('BRIGHT')
    afw_image.Mask.addMaskPlane('BRIGHT_EXPANDED')
    bright = afw_image.Mask.getPlaneBitMask('BRIGHT')
    bright_expanded = afw_image.Mask.getPlaneBitMask('BRIGHT_EXPANDED')

    wcs = mbobs[0][0].exposure.getWcs()

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
