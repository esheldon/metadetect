import numpy as np
import ngmix
import pytest


def test_measure_mfrac_neg_bbox():
    pytest.importorskip("meds")
    from ..mfrac import measure_mfrac

    rng = np.random.RandomState(seed=100)
    cen = (201-1)/2
    mfrac = rng.uniform(size=(201, 201), low=0.2, high=0.8)
    x = rng.uniform(size=3, low=50, high=150)
    y = rng.uniform(size=3, low=50, high=150)
    box_sizes = np.array([32, -9990, 0], dtype=np.int32)
    fwhm = 1.2

    obs = ngmix.Observation(
        image=np.zeros_like(mfrac),
        bmask=np.zeros_like(mfrac, dtype=np.int32),
        weight=np.ones_like(mfrac),
        jacobian=ngmix.DiagonalJacobian(scale=0.263, row=cen, col=cen)
    )
    obs.psf = obs.copy()

    mes = measure_mfrac(
        mfrac=mfrac,
        x=x,
        y=y,
        box_sizes=box_sizes,
        obs=obs,
        fwhm=fwhm,
    )
    assert mes[0] > 0.4 and mes[0] < 0.6
    assert mes[1] == 1.0
    assert mes[2] == 1.0
