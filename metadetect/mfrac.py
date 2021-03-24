import ngmix
import numpy as np

from .shearpos import DEFAULT_STEP
from .detect import MEDSInterface
from .defaults import BMASK_EDGE


def measure_mfrac(
    *,
    mfrac,
    cat,
    shear_str,
    obs,
    fwhm,
    step=DEFAULT_STEP,
):
    if fwhm is None:
        fwhm = 1.2

    obs = obs.copy()
    obs.set_image(mfrac)

    if shear_str == '1p':
        g1, g2 = step, 0.0
    elif shear_str == '1m':
        g1, g2 = -step, 0.0
    elif shear_str == '2p':
        g1, g2 = 0.0, step
    elif shear_str == '2m':
        g1, g2 = 0.0, -step
    elif shear_str == 'noshear':
        g1, g2 = 0.0, 0.0
    else:
        raise ValueError('Can only convert 1p,1m,2p,2m to a shear!')

    gauss_wgt = ngmix.GMixModel(
        [0, 0, g1, g2, ngmix.moments.fwhm_to_T(fwhm), 1],
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
