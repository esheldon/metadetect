"""
Code to do metacal with lsst exposures
"""
import numpy as np
from ngmix.metacal.metacal import _get_gauss_target_psf
import galsim
import lsst.afw.image as afw_image
from .util import (
    get_integer_center, get_jacobian, get_stack_kernel_psf, get_mbexp,
)

DEFAULT_TYPES = ['noshear', '1p', '1m']
INTERP = 'lanczos15'
STEP = 0.01


def get_metacal_mbexps_fixnoise(mbexp, noise_mbexp, types=None):
    """
    Get metacal MultibandExposures with fixed noise

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposure data
    noise_mbexp: lsst.afw.image.MultibandExposure
        The exposure data with pure noise
    types: list, optional
        The metacal types, e.g. ('noshear', '1p', '1m')

    Returns
    -------
    mdict, noise_mdict
        dicts keyed by type, holding exposures
    """

    mdict = get_metacal_mbexps(mbexp=mbexp, types=types)
    noise_mdict = get_metacal_mbexps(mbexp=noise_mbexp, types=types, rot=True)
    for shear_type in mdict:
        for exp, nexp in zip(mdict[shear_type], noise_mdict[shear_type]):
            exp.image.array[:, :] += nexp.image.array[:, :]
            exp.variance.array[:, :] *= 2

    return mdict, noise_mdict


def get_metacal_mbexps(mbexp, types=None, rot=False):
    """
    Get metacal MultibandExposures

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposure data
    types: list, optional
        The metacal types, e.g. ('noshear', '1p', '1m')
    rot: bool, optional
        If set to True, rotate before shearing, then rotate back.

    Returns
    -------
    mdict
        dict keyed by type, holding exposures
    """

    if types is None:
        types = DEFAULT_TYPES

    mdict_with_explists = {}
    for shear_type in types:
        mdict_with_explists[shear_type] = []

    for iband, band in enumerate(mbexp.bands):
        exp = mbexp[band]

        this_mdict = get_metacal_exps(exp, types=types, rot=rot)

        for shear_type in this_mdict:
            exp = this_mdict[shear_type]
            mdict_with_explists[shear_type].append(exp)

    # now build the mbexp
    mdict = {}
    for shear_type in types:
        # this properly copies over the wcs and filter label
        mdict[shear_type] = get_mbexp(mdict_with_explists[shear_type])

    return mdict


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

    gwcs = get_galsim_jacobian_wcs(exp=exp, cen=cen)

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

    gauss_psf = _get_gauss_target_psf(psf_int, flux=psf_flux)

    dilation = 1.0 + 2.0*STEP
    psf_dilated = gauss_psf.dilate(dilation)

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
    """
    Convert a shear type string to a galsim Shear

    Parameters
    ----------
    shear_type: str
        One of ('1p', '1m', '2p', '1m')

    Returns
    -------
    galsim.Shear
    """
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
    """
    Get a galsim jacobian wcs object from the input exposure's
    wcs and the input location

    Parameters
    -----------
    exp: lsst.afw.image.Exposure*
        An lsst exposure object
    cen: lsst.afw.geom.Point2D
        The point in the image at which to construct the wcs

    Returns
    -------
    galsim.JacobianWCS
    """
    jac = get_jacobian(exp=exp, cen=cen)
    return jac.get_galsim_wcs()


def get_psf_kernel_image(exp, cen):
    """
    Reconstruct a kernel image for the input exposure at the specified
    location

    Parameters
    -----------
    exp: lsst.afw.image.Exposure*
        An lsst exposure object
    cen: lsst.afw.geom.Point2D
        The point in the image at which to construct the wcs

    Returns
    -------
    image as a numpy array
    """
    psf_obj = exp.getPsf()
    return psf_obj.computeKernelImage(cen).array
