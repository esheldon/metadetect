"""
Code to do metacal with lsst exposures
"""
import numpy as np
from ngmix.metacal.metacal import _get_gauss_target_psf
import galsim
import lsst.afw.image as afw_image
from .util import get_integer_center, get_jacobian, get_stack_kernel_psf

DEFAULT_TYPES = ['noshear', '1p', '1m']
INTERP = 'lanczos15'
STEP = 0.01


def get_metacal_exps_fixnoise(exp, noise_exp, types=None):
    """
    Get metacal exposures with fixed noise

    Parameters
    ----------
    exp: lsst.afw.image.Exposure
        The exposure data
    noise_exp: lsst.afw.image.Exposure
        The exposure data with pure noise
    types: list, optional
        The metacal types, e.g. ('noshear', '1p', '1m')

    Returns
    -------
    dict keyed by type, holding exposures
    """
    if types is None:
        types = DEFAULT_TYPES

    mdict = get_metacal_exps(exp, types=types)
    noise_mdict = get_metacal_exps(noise_exp, types=types, rot=True)

    for shear_type in types:
        exp = mdict[shear_type]
        nexp = noise_mdict[shear_type]

        exp.image.array[:, :] += nexp.image.array[:, :]
        exp.variance.array[:, :] *= 2

    return mdict, noise_mdict


def get_metacal_exps(exp, types=None, rot=False):
    """
    Get metacal exposures

    Parameters
    ----------
    exp: lsst.afw.image.Exposure
        The exposure data
    types: list, optional
        The metacal types, e.g. ('noshear', '1p', '1m')
    rot: bool, optional
        If set to True, rotate before shearing, then rotate back.

    Returns
    -------
    dict keyed by type, holding exposures
    """

    if types is None:
        types = DEFAULT_TYPES

    cen, _ = get_integer_center(exp.getWcs(), exp.getBBox(), as_double=True)

    # wcs = exp.getWcs()

    gwcs = get_galsim_jacobian_wcs(exp=exp, cen=cen)
    pixel = gwcs.toWorld(galsim.Pixel(scale=1))
    pixel_inv = galsim.Deconvolve(pixel)

    psf_image_array = get_psf_kernel_image(exp=exp, cen=cen)
    psf_flux = psf_image_array.sum()

    eimage = exp.image.array.copy()
    if rot:
        eimage = np.rot90(eimage, k=1)

    # force double, *it matters*.  There are clear artifacts
    # present when comparing to the regular metacal if float32 is
    # used
    image = galsim.ImageD(eimage, wcs=gwcs)
    psf_image = galsim.ImageD(psf_image_array, wcs=gwcs)

    # interpolated psf image and its deconvolution
    psf_int = galsim.InterpolatedImage(psf_image, x_interpolant=INTERP)

    # deconvolved galaxy image, psf+pixel removed
    image_int_nopsf = galsim.Convolve(
        galsim.InterpolatedImage(image, x_interpolant=INTERP),
        galsim.Deconvolve(psf_int),
    )

    # TODO we may not need to deconvolve the pixel from the psf
    # interpolated psf deconvolved from pixel.  This is what
    # we dilate, shear, etc and reconvolve the image by
    psf_int_nopix = galsim.Convolve([psf_int, pixel_inv])

    gauss_psf = _get_gauss_target_psf(psf_int_nopix, flux=psf_flux)

    dilation = 1.0 + 2.0*STEP
    psf_dilated_nopix = gauss_psf.dilate(dilation)
    psf_dilated = galsim.Convolve(psf_dilated_nopix, pixel)

    psf_dilated_image = psf_image.copy()
    psf_dilated.drawImage(image=psf_dilated_image, method='no_pixel')

    mdict = {}
    for shear_type in types:
        mdict[shear_type] = _get_metacal_exp(
            exp=exp, image=image, image_int_nopsf=image_int_nopsf,
            psf_dilated=psf_dilated, psf_dilated_image=psf_dilated_image,
            shear_type=shear_type, rot=rot,
        )

    return mdict


def _get_metacal_exp(
    exp, image, image_int_nopsf, psf_dilated, psf_dilated_image, shear_type,
    rot,
):
    sexp = afw_image.ExposureD(exp, deep=True)

    if shear_type == 'noshear':
        obj = image_int_nopsf
    else:
        shear = get_shear_from_type(shear_type)
        obj = image_int_nopsf.shear(shear)

    shimage = image.copy()
    galsim.Convolve(obj, psf_dilated).drawImage(
        image=shimage, method='no_pixel',
    )

    simage = shimage.array
    if rot:
        simage = np.rot90(simage, k=3)

    sexp.image.array[:, :] = simage

    stack_psf = get_stack_kernel_psf(psf_dilated_image.array)
    sexp.setPsf(stack_psf)
    return sexp


def get_shear_from_type(shear_type):

    if shear_type == '1p':
        shear = galsim.Shear(g1=STEP)
    elif shear_type == '1m':
        shear = galsim.Shear(g1=-STEP)
    elif shear_type == '2p':
        shear = galsim.Shear(g2=STEP)
    elif shear_type == '2m':
        shear = galsim.Shear(g2=-STEP)
    else:
        raise ValueError('shear type should be noshear, 1p, 1m, 2p, 2m')

    return shear


def get_galsim_jacobian_wcs(exp, cen):
    jac = get_jacobian(exp=exp, cen=cen)
    return jac.get_galsim_wcs()


def get_psf_kernel_image(exp, cen):
    psf_obj = exp.getPsf()
    return psf_obj.computeKernelImage(cen).array
