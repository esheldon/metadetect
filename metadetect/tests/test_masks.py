import numpy as np
import ngmix

import pytest

from ..masks import (
    _intersects,
    apply_mask_mbobs,
    apply_mask_mfrac,
    apply_mask_bit_mask,
)


@pytest.mark.parametrize('row,col,radius_pixels,nrows,ncols,yes', [
    # basic
    (0, 0, 10, 10, 10, True),
    (-1000, 0, 10, 10, 10, False),
    (1000, 0, 10, 10, 10, False),
    (0, -1000, 10, 10, 10, False),
    (0, 1000, 10, 10, 10, False),
    # edge cases
    (-10, 0, 10, 10, 10, False),
    (19, 0, 10, 10, 10, False),
    (0, -10, 10, 10, 10, False),
    (0, 19, 10, 10, 10, False),
    # partial
    (2, 5, 10, 7, 7, True),
])
def test_intersects(row, col, radius_pixels, nrows, ncols, yes):
    if yes:
        assert _intersects(row, col, radius_pixels, nrows, ncols)
    else:
        assert not _intersects(row, col, radius_pixels, nrows, ncols)


@pytest.mark.parametrize('symmetrize', [False, True])
def test_apply_mask_mfrac(symmetrize):
    dims = (100, 100)
    mfrac = np.zeros(dims, dtype='f8')
    mfrac[0, 0] = 1
    mask_catalog = np.array(
        [(99, 99, 10)],
        dtype=[('x', 'f8'), ('y', 'f8'), ('radius_pixels', 'f8')],
    )
    apply_mask_mfrac(mfrac, mask_catalog, symmetrize=symmetrize)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(mfrac)
        import pdb
        pdb.set_trace()

    assert mfrac[0, 0] == 1
    assert np.all(mfrac[98:, 98:] == 1)

    if symmetrize:
        assert np.all(mfrac[:2, 98:] == 1)
    else:
        assert np.all(mfrac[:2, 98:] == 0)


@pytest.mark.parametrize('symmetrize', [False, True])
def test_apply_mask_bit_mask(symmetrize):
    dims = (100, 100)
    bmask = np.zeros(dims, dtype='i4')
    maskflags = 2**2
    bmask[0, 0] |= maskflags
    bmask[98:, 98:] |= 2**3
    mask_catalog = np.array(
        [(99, 99, 10)],
        dtype=[('x', 'f8'), ('y', 'f8'), ('radius_pixels', 'f8')],
    )
    apply_mask_bit_mask(bmask, mask_catalog, maskflags, symmetrize=symmetrize)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(bmask)
        import pdb
        pdb.set_trace()

    assert bmask[0, 0] == maskflags
    assert np.all((bmask[98:, 98:] & maskflags) != 0)
    assert np.all((bmask[98:, 98:] & 2**3) != 0)

    if symmetrize:
        assert np.all((bmask[:2, 98:] & maskflags) != 0)
    else:
        assert np.all((bmask[:2, 98:] & maskflags) == 0)


@pytest.mark.parametrize('symmetrize', [False, True])
def test_apply_mask_mbobs(symmetrize):
    dims = (100, 100)
    maskflags = 2**2
    mask_catalog = np.array(
        [(99, 99, 10)],
        dtype=[('x', 'f8'), ('y', 'f8'), ('radius_pixels', 'f8')],
    )

    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    mbobs.append(obslist)

    for i in range(2):
        bmask = np.zeros(dims, dtype='i4')
        bmask[i, i] |= maskflags
        bmask[98:, 98:] |= 2**(i+3)

        mfrac = np.zeros(dims, dtype='f4')
        mfrac[i, i] = 1

        obs = ngmix.Observation(
            image=np.zeros(dims, dtype='i4') + (i + 6),
            weight=np.zeros(dims, dtype='i4') + (i + 7),
            bmask=bmask,
            mfrac=mfrac,
        )
        obslist.append(obs)

    apply_mask_mbobs(mbobs, mask_catalog, maskflags, symmetrize=symmetrize)

    for i, obs in enumerate(mbobs[0]):
        io = (i+1) % 2

        wgt = obs.weight
        assert np.all(wgt[98:, 98:] == 0)
        if symmetrize:
            assert np.all(wgt[:2, 98:] == 0)
        else:
            assert np.all(wgt[:2, 98:] == (i+7))

        mfrac = obs.mfrac
        assert mfrac[i, i] == 1
        assert mfrac[io, io] == 0
        assert np.all(mfrac[98:, 98:] == 1)

        if symmetrize:
            assert np.all(mfrac[:2, 98:] == 1)
        else:
            assert np.all(mfrac[:2, 98:] == 0)

        bmask = obs.bmask
        assert bmask[i, i] == maskflags
        assert bmask[io, io] == 0
        assert np.all((bmask[98:, 98:] & maskflags) != 0)
        assert np.all((bmask[98:, 98:] & 2**(i+3)) != 0)

        if symmetrize:
            assert np.all((bmask[:2, 98:] & maskflags) != 0)
        else:
            assert np.all((bmask[:2, 98:] & maskflags) == 0)
