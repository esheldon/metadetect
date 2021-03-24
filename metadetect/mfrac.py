import ngmix
import numpy as np

from .shearpos import DEFAULT_STEP
from .detect import MEDSInterface
from .defaults import BMASK_EDGE


def measure_mfrac(
    *,
    mfrac,
    cat,
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
    m = MEDSInterface(
        obs,
        np.zeros_like(mfrac, dtype=np.int32),
        cat,
    )
    mfracs = []
    for i in range(cat.shape[0]):
        obs = m.get_obs(i, 0)
        wgt = obs.weight.copy()
        msk = (obs.bmask & BMASK_EDGE) != 0
        wgt[msk] = 0
        wgt[~msk] = 1
        obs.set_weight(wgt)

        stats = gauss_wgt.get_weighted_sums(
            obs,
            obs.image.shape[0] * obs.jacobian.get_scale() / 2,
        )
        mfracs.append(stats["sums"][5] / stats["wsum"])

    return np.array(mfracs)
