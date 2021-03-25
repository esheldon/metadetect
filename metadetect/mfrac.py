import ngmix
import numpy as np

from .shearpos import DEFAULT_STEP
from .detect import CatalogMEDSifier
from .defaults import BMASK_EDGE


def measure_mfrac(
    *,
    mfrac,
    x,
    y,
    box_sizes,
    obs,
    fwhm,
    step=DEFAULT_STEP,
):
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
        mfracs.append(stats["sums"][5] / stats["wsum"])

    return np.array(mfracs)
