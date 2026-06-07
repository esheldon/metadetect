import os
import sys
import numpy as np
import pytest
import time

import logging
from metadetect.lsst.measure import get_detect_and_deblend_task
from metadetect.lsst import util
from metadetect.lsst import vis
from metadetect.lsst.model_subtractor import ModelSubtractor
from metadetect.lsst.mbobs_extractor import MBObsExtractor
from lsst.pex.exceptions import LengthError

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
)


def make_lsst_sim(rng, bands=['g', 'r', 'i']):
    import descwl_shear_sims

    coadd_dim = 251

    galaxy_catalog = descwl_shear_sims.galaxies.WLDeblendGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=20,
    )

    psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = descwl_shear_sims.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        g1=0.02,
        g2=0.00,
        psf=psf,
        bands=bands,
    )
    return sim_data


def do_coadding(rng, sim_data, nowarp=True):
    from descwl_coadd.coadd import make_coadd
    from descwl_coadd.coadd_nowarp import make_coadd_nowarp

    bands = list(sim_data['band_data'].keys())

    if nowarp:
        coadd_data_list = [
            make_coadd_nowarp(
                exp=sim_data['band_data'][band][0],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                remove_poisson=False,
            )
            for band in bands
        ]
    else:
        coadd_data_list = [
            make_coadd(
                exps=sim_data['band_data'][band],
                psf_dims=sim_data['psf_dims'],
                rng=rng,
                coadd_wcs=sim_data['coadd_wcs'],
                coadd_bbox=sim_data['coadd_bbox'],
                remove_poisson=False,
            )
            for band in bands
        ]

    return util.extract_multiband_coadd_data(coadd_data_list)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_lsst_model_subtractor_smoke(seed=225, ntrial=1, show=False):
    rng = np.random.RandomState(seed)

    tm = 0.0
    for i in range(ntrial):
        sim_data = make_lsst_sim(rng)
        data = do_coadding(rng=rng, sim_data=sim_data)
        mbexp = data['mbexp']

        if show:
            vis.show_mbexp(mbexp)

        tm0 = time.time()
        dbtask = get_detect_and_deblend_task(rng=rng, deblender='scarlet')
        sources, detecp, model_data = dbtask.run(
            mbexp=mbexp,
            show=show,
        )

        subtractor = ModelSubtractor(
            mbexp=mbexp,
            sources=sources,
            model_data=model_data,
        )

        tm += time.time() - tm0

        if show:
            stamp_size = 49
            # vis.show_mbexp(subtractor.mbexp)
            vis.show_mbexp_mosaic(
                [mbexp, subtractor.get_full_model(), subtractor.mbexp],
                labels=['orig', 'model', 'subtracted'],
            )

            for i, source in enumerate(subtractor.children()):
                source_id = source.getId()
                try:
                    with subtractor.add_source(source_id):
                        stamp = subtractor.get_stamp(
                            source_id,
                            stamp_size=stamp_size,
                        )
                        model = subtractor.get_model(
                            source_id,
                            stamp_size=stamp_size,
                        )
                        vis.show_mbexp_mosaic(
                            [subtractor.mbexp, stamp, model],
                            labels=[
                                f'inserted {i}',
                                f'inserted stamp {i}',
                                f'model {i}',
                            ],
                        )
                except LengthError:
                    pass

    print('time per:', tm / ntrial)


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason='simulation input data is not present',
)
def test_lsst_mbobs_extractor_smoke(seed=225, show=False):

    rng = np.random.RandomState(seed)
    sim_data = make_lsst_sim(rng)

    data = do_coadding(rng=rng, sim_data=sim_data)
    mbexp = data['mbexp']

    if show:
        vis.show_mbexp(mbexp)

    dbtask = get_detect_and_deblend_task(rng=rng, deblender='sdss')
    sources, detecp, model_data = dbtask.run(
        mbexp=mbexp,
        show=show,
    )

    _ = MBObsExtractor(
        mbexp=mbexp,
        sources=sources,
    )


if __name__ == '__main__':
    test_lsst_model_subtractor_smoke(seed=225, show=True)
    # test_lsst_model_subtractor_smoke(seed=225, ntrial=20)
