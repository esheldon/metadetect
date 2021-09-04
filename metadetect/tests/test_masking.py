import numpy as np
import ngmix

import pytest

from ..masking import (
    _intersects,
    _ap_kern_kern,
    _do_apodization_mask,
    _make_foreground_apodization_mask,
    _do_mask_foreground,
    _make_foreground_bmask,
    apply_foreground_masking_corrections,
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


@pytest.mark.parametrize("fac,val", [
    (1, 1),
    (0, 1),
    (-3, 0.5),
    (-6, 0),
    (-7, 0),
])
def test_ap_kern_kern(fac, val):
    h = 2.0
    m = 10.0
    assert np.allclose(_ap_kern_kern(m + fac*h, m, h), val)


def test_do_apodization_mask_all_masked():
    ap_mask = np.ones((10, 13))
    rows = np.array([0, 3])
    cols = rows = np.array([0, 5])
    radius_pixels = np.array([100, 10])

    _do_apodization_mask(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        ap_mask=ap_mask,
        ap_rad=1.5,
    )

    assert np.all(ap_mask == 0)


def test_do_apodization_mask_half_masked():
    ap_mask = np.ones((10, 13))
    rows = np.array([0])
    cols = np.array([6])
    radius_pixels = np.array([5])

    _do_apodization_mask(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        ap_mask=ap_mask,
        ap_rad=0.1,
    )

    assert np.all(ap_mask[6:, :] == 1)
    assert not np.all(ap_mask[:6, :] == 1)
    assert np.all(ap_mask[0:2, 6:8] == 0)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(ap_mask)
        import pdb
        pdb.set_trace()


def test_make_foreground_apodization_mask():
    ap_mask = _make_foreground_apodization_mask(
        xm=np.array([6], dtype='f8'),
        ym=np.array([0], dtype='f8'),
        rm=np.array([5], dtype='f8'),
        dims=(10, 13),
        symmetrize=False,
        ap_rad=0.1
    )

    assert np.all(ap_mask[6:, :] == 1)
    assert not np.all(ap_mask[:6, :] == 1)
    assert np.all(ap_mask[0:2, 6:8] == 0)


def test_make_foreground_apodization_mask_symmetrize():
    ap_mask = _make_foreground_apodization_mask(
        xm=np.array([6], dtype='f8'),
        ym=np.array([0], dtype='f8'),
        rm=np.array([5], dtype='f8'),
        dims=(10, 10),
        symmetrize=False,
        ap_rad=0.1
    )

    ap_mask_sym = _make_foreground_apodization_mask(
        xm=np.array([6], dtype='f8'),
        ym=np.array([0], dtype='f8'),
        rm=np.array([5], dtype='f8'),
        dims=(10, 10),
        symmetrize=True,
        ap_rad=0.1
    )

    msk = ap_mask == 0
    assert np.allclose(ap_mask[msk], ap_mask_sym[msk])

    msk = ap_mask_sym == 1
    assert np.allclose(ap_mask[msk], ap_mask_sym[msk])

    rot_ap_mask = np.rot90(ap_mask)
    msk = rot_ap_mask == 0
    assert np.allclose(ap_mask_sym[msk], rot_ap_mask[msk])

    assert np.all(ap_mask[0:2, 6:8] == 0)

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(ap_mask_sym)
        import pdb
        pdb.set_trace()


def test_do_mask_foreground_all_masked():
    flag = 2**5
    bmask = np.zeros((10, 13), dtype=np.int32)
    rows = np.array([0, 3])
    cols = rows = np.array([0, 5])
    radius_pixels = np.array([100, 10])
    bmask[4, 5] |= 2**3

    _do_mask_foreground(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        bmask=bmask,
        flag=flag,
    )

    assert np.all((bmask & flag) != 0)
    assert (bmask[4, 5] & 2**3) != 0
    assert (bmask[1, 2] & 2**3) == 0


def test_do_mask_foreground_half_masked():
    flag = 2**5
    bmask = np.zeros((10, 13), dtype=np.int32)
    rows = np.array([0])
    cols = np.array([6])
    radius_pixels = np.array([5])
    bmask[4, 5] |= 2**3

    _do_mask_foreground(
        rows=rows,
        cols=cols,
        radius_pixels=radius_pixels,
        bmask=bmask,
        flag=flag,
    )

    assert np.all((bmask[6:, 6:] & flag) == 0)
    assert np.all((bmask[0:2, 6:8] & flag) != 0)
    assert (bmask[4, 5] & 2**3) != 0
    assert (bmask[1, 2] & 2**3) == 0


def test_make_foreground_bmask():
    flag = 2**7

    bmask = _make_foreground_bmask(
        xm=np.array([6], dtype='f8'),
        ym=np.array([0], dtype='f8'),
        rm=np.array([5], dtype='f8'),
        dims=(10, 13),
        symmetrize=False,
        mask_bit_val=flag,
    )

    assert np.all((bmask[6:, 6:] & flag) == 0)
    assert np.all((bmask[0:2, 6:8] & flag) != 0)


def test_make_foreground_bmask_symmetrize():
    flag = 2**9

    bmask = _make_foreground_bmask(
        xm=np.array([6], dtype='f8'),
        ym=np.array([0], dtype='f8'),
        rm=np.array([5], dtype='f8'),
        dims=(10, 10),
        mask_bit_val=flag,
        symmetrize=False,
    )

    bmask_sym = _make_foreground_bmask(
        xm=np.array([6], dtype='f8'),
        ym=np.array([0], dtype='f8'),
        rm=np.array([5], dtype='f8'),
        dims=(10, 10),
        mask_bit_val=flag,
        symmetrize=True,
    )

    msk = bmask > 0
    assert np.all((bmask[msk] & bmask_sym[msk]) != 0)

    rot_bmask = np.rot90(bmask)
    msk = rot_bmask > 0
    assert np.all((rot_bmask[msk] & bmask_sym[msk]) != 0)

    assert np.all((bmask_sym[0:2, 6:8] & flag) != 0)


@pytest.mark.parametrize("msk_exp_rad", [0, 4])
def test_apply_foreground_masking_corrections_interp(msk_exp_rad):
    nband = 2
    seed = 10
    dims = (13, 13)

    def _make_mbobs():
        mbobs = ngmix.MultiBandObsList()
        rng = np.random.RandomState(seed=seed)
        for _ in range(nband):
            image = rng.uniform(size=dims)
            noise = rng.uniform(size=dims)
            obs = ngmix.Observation(
                image=image,
                noise=noise,
                weight=rng.uniform(size=dims),
                bmask=np.zeros(dims, dtype=np.int32),
                ormask=np.zeros(dims, dtype=np.int32),
            )
            obs.mfrac = rng.uniform(size=dims)
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs

    res = {}
    for method in ['interp', 'interp-noise']:
        mbobs = _make_mbobs()
        apply_foreground_masking_corrections(
            mbobs=mbobs,
            xm=np.array([6]),
            ym=np.array([0]),
            rm=np.array([5-msk_exp_rad]),
            method=method,
            mask_expand_rad=msk_exp_rad,
            mask_bit_val=2**3,
            expand_mask_bit_val=2**4,
            interp_bit_val=2**5,
            symmetrize=False,
            ap_rad=1,
            iso_buff=1,
            rng=np.random.RandomState(seed=11)
        )
        res[method] = mbobs

    for method in ['interp', 'interp-noise']:
        mbobs = res[method]
        rng = np.random.RandomState(seed=seed)
        for obslist in mbobs:
            for obs in obslist:
                msk = (obs.bmask & 2**3) != 0
                image = rng.uniform(size=dims)
                noise = rng.uniform(size=dims)
                # we need to match these calls to the ones above
                rng.uniform(size=dims)
                rng.uniform(size=dims)

                assert np.all(obs.mfrac[msk] == 1)
                assert np.all(obs.weight[msk] == 0)
                assert np.all((obs.bmask[msk] & 2**5) != 0)

                assert np.all(image[msk] != obs.image[msk])
                assert np.all(image[~msk] == obs.image[~msk])
                assert np.all(noise[msk] != obs.noise[msk])
                assert np.all(noise[~msk] == obs.noise[~msk])

                msk = (obs.bmask & 2**4) != 0
                if msk_exp_rad > 0:
                    assert np.sum(msk) > 0

                    assert not np.all(obs.mfrac[msk] == 1)
                    assert not np.all(obs.weight[msk] == 0)
                    assert not np.all((obs.bmask[msk] & 2**5) != 0)

                    assert not np.all(image[msk] != obs.image[msk])
                    assert np.all(image[~msk] == obs.image[~msk])
                    assert not np.all(noise[msk] != obs.noise[msk])
                    assert np.all(noise[~msk] == obs.noise[~msk])
                else:
                    assert np.sum(msk) == 0

    for i in range(len(res['interp'])):
        if msk_exp_rad > 0:
            assert np.array_equal(
                res['interp'][i][0].image, res['interp-noise'][i][0].image
            )
        else:
            assert not np.array_equal(
                res['interp'][i][0].image, res['interp-noise'][i][0].image
            )


@pytest.mark.parametrize('method', ['interp', 'interp-noise'])
def test_apply_foreground_masking_corrections_interp_all(method):
    nband = 2
    seed = 10
    dims = (13, 13)
    mbobs = ngmix.MultiBandObsList()
    rng = np.random.RandomState(seed=seed)
    for _ in range(nband):
        image = rng.uniform(size=dims)
        noise = rng.uniform(size=dims)
        obs = ngmix.Observation(
            image=image,
            noise=noise,
            weight=rng.uniform(size=dims),
            bmask=np.zeros(dims, dtype=np.int32),
            ormask=np.zeros(dims, dtype=np.int32),
        )
        obs.mfrac = rng.uniform(size=dims)
        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    apply_foreground_masking_corrections(
        mbobs=mbobs,
        xm=np.array([6]),
        ym=np.array([0]),
        rm=np.array([100]),
        method=method,
        mask_expand_rad=0,
        mask_bit_val=2**3,
        expand_mask_bit_val=2**4,
        interp_bit_val=2**5,
        symmetrize=False,
        ap_rad=1,
        iso_buff=1,
        rng=np.random.RandomState(seed=11)
    )

    rng = np.random.RandomState(seed=seed)
    for obslist in mbobs:
        for obs in obslist:
            assert np.all(obs.mfrac == 1)
            assert np.all(obs.weight == 0)
            assert np.all((obs.bmask & 2**3) != 0)


# @pytest.mark.parametrize("msk_exp_rad", [0, 4])
# def test_mask_gaia_stars_apodize(msk_exp_rad):
#     nband = 2
#     seed = 10
#     start_row = 1012
#     start_col = 4513
#     gaia_stars = np.array(
#         [(6+start_col, 0+start_row, 5-msk_exp_rad)],
#         dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
#     )
#     dims = (13, 13)
#     config = dict(
#         symmetrize=False,
#         apodize={"ap_rad": 0.5},
#         mask_expand_rad=msk_exp_rad,
#     )
#     mbobs = ngmix.MultiBandObsList()
#     rng = np.random.RandomState(seed=seed)
#     for _ in range(nband):
#         image = rng.uniform(size=dims)
#         noise = rng.uniform(size=dims)
#         obs = ngmix.Observation(
#             image=image,
#             noise=noise,
#             weight=rng.uniform(size=dims),
#             bmask=np.zeros(dims, dtype=np.int32),
#             ormask=np.zeros(dims, dtype=np.int32),
#             meta={"orig_start_row": start_row, "orig_start_col": start_col},
#         )
#         obs.mfrac = rng.uniform(size=dims)
#         obslist = ngmix.ObsList()
#         obslist.append(obs)
#         mbobs.append(obslist)
#
#     mask_gaia_stars(mbobs, gaia_stars, config)
#
#     rng = np.random.RandomState(seed=seed)
#     for obslist in mbobs:
#         for obs in obslist:
#             msk = (obs.bmask & BMASK_GAIA_STAR) != 0
#             image = rng.uniform(size=dims)
#             noise = rng.uniform(size=dims)
#             # we need to match these calls to the ones above
#             rng.uniform(size=dims)
#             rng.uniform(size=dims)
#
#             assert np.all(obs.mfrac[msk] == 1)
#             assert np.all(obs.weight[msk] == 0)
#
#             assert np.all(image[msk] != obs.image[msk])
#             assert np.all(image[~msk] == obs.image[~msk])
#             assert np.all(noise[msk] != obs.noise[msk])
#             assert np.all(noise[~msk] == obs.noise[~msk])
#
#             msk = (obs.bmask & BMASK_EXPAND_GAIA_STAR) != 0
#             if msk_exp_rad > 0:
#                 assert np.sum(msk) > 0
#
#                 assert not np.all(obs.mfrac[msk] == 1)
#                 assert not np.all(obs.weight[msk] == 0)
#
#                 assert not np.all(image[msk] != obs.image[msk])
#                 assert np.all(image[~msk] == obs.image[~msk])
#                 assert not np.all(noise[msk] != obs.noise[msk])
#                 assert np.all(noise[~msk] == obs.noise[~msk])
#             else:
#                 assert np.sum(msk) == 0
#
#
# def test_mask_gaia_stars_apodize_all():
#     nband = 2
#     seed = 10
#     start_row = 1012
#     start_col = 4513
#     gaia_stars = np.array(
#         [(6+start_col, 0+start_row, 100)],
#         dtype=[('x', 'f8'), ('y', '<f4'), ('radius_pixels', '>f4')]
#     )
#     dims = (13, 13)
#     config = dict(
#         symmetrize=False,
#         apodize={"ap_rad": 0.5},
#         mask_expand_rad=0,
#     )
#     mbobs = ngmix.MultiBandObsList()
#     rng = np.random.RandomState(seed=seed)
#     for _ in range(nband):
#         image = rng.uniform(size=dims)
#         noise = rng.uniform(size=dims)
#         obs = ngmix.Observation(
#             image=image,
#             noise=noise,
#             weight=rng.uniform(size=dims),
#             bmask=np.zeros(dims, dtype=np.int32),
#             ormask=np.zeros(dims, dtype=np.int32),
#             meta={"orig_start_row": start_row, "orig_start_col": start_col},
#         )
#         obs.mfrac = rng.uniform(size=dims)
#         obslist = ngmix.ObsList()
#         obslist.append(obs)
#         mbobs.append(obslist)
#
#     mask_gaia_stars(mbobs, gaia_stars, config)
#
#     rng = np.random.RandomState(seed=seed)
#     for obslist in mbobs:
#         for obs in obslist:
#             assert np.all(obs.mfrac == 1)
#             assert np.all(obs.weight == 0)
#             assert np.all((obs.bmask & BMASK_GAIA_STAR) != 0)
#             assert np.all(obs.image == 0)
#             assert np.all(obs.noise == 0)
