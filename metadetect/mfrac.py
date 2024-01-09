import ngmix
import numpy as np

from .defaults import BMASK_EDGE


def measure_mfrac(
    *,
    mfrac,
    x,
    y,
    box_sizes,
    obs,
    fwhm,
):
    """Measure a Gaussian-weighted average of an image.

    This function is meant to be used with images that represent the fraction
    of single-epoch images that are masked in each pixel of a coadd. It
    computes a Gaussian-weighted average of the image at a list of locations.

    Parameters
    ----------
    mfrac : np.ndarray
        The input image with which to compute the weighted averages.
    x : np.ndarray
        The input x/col values for the positions at which to compute the
        weighted average.
    y : np.ndarray
        The input y/row values for the positions at which to compute the
        weighted average.
    box_sizes : np.ndarray
        The size of the stamp to use to measure the weighted average. Should be
        big enough to hold 2 * `fwhm`.
    obs : ngmix.Observation
        An observation that holds the weight maps, WCS Jacobian, etc
        corresponding to `mfrac`.
    fwhm : float or None
        The FWHM of the Gaussian aperture in arcseconds. If None, a default
        of 1.2 is used.

    Returns
    -------
    mfracs : np.ndarray
        The weighted averages at each input location.
    """
    from .detect import CatalogMEDSifier

    if fwhm is None:
        fwhm = 1.2

    obs = obs.copy()
    obs.set_image(mfrac)

    gauss_wgt = ngmix.GMixModel(
        [0, 0, 0, 0, ngmix.moments.fwhm_to_T(fwhm), 1],
        'gauss',
    )
    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    mbobs.append(obslist)
    obslist.append(obs)
    m = CatalogMEDSifier(mbobs, x, y, box_sizes).get_meds(0)
    mfracs = []
    for i in range(x.shape[0]):
        try:
            if box_sizes[i] > 0:
                obs = m.get_obs(i, 0)
                wgt = obs.weight.copy()
                msk = (obs.bmask & BMASK_EDGE) != 0
                wgt[msk] = 0
                wgt[~msk] = 1
                obs.set_weight(wgt)

                stats = gauss_wgt.get_weighted_sums(
                    obs,
                    fwhm * 2,
                )
                # this is the weighted average in the image using the
                # Gaussian as the weight.
                mfracs.append(stats["sums"][5] / stats["wsum"])
            else:
                mfracs.append(1.0)
        except ngmix.GMixFatalError:
            mfracs.append(1.0)

    return np.array(mfracs)
