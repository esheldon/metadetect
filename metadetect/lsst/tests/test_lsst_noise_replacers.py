import sys
import numpy as np
import logging
import pytest

from metadetect.lsst import util
from lsst.utils import getPackageDir

try:
    getPackageDir('descwl_shear_sims')
    skip_tests_on_simulations = False
except LookupError:
    skip_tests_on_simulations = True

logging.basicConfig(stream=sys.stdout, level=logging.WARN)


def make_lsst_sim(rng):
    import descwl_shear_sims
    coadd_dim = 200
    buff = 20
    # coadd_dim = 351
    # buff = 50

    # the EDGE region is 5 pixels wide but, give a bit more space because the
    # sky sub seems to fail with a lsst.pex.exceptions.wrappers.LengthError,
    # presumably due to an object near the edge

    bands = ['r', 'i', 'z']

    galaxy_catalog = descwl_shear_sims.galaxies.FixedGalaxyCatalog(
        rng=rng,
        coadd_dim=coadd_dim,
        buff=buff,
        layout='random',
        mag=22,
        hlr=0.5,
    )

    psf = descwl_shear_sims.psfs.make_fixed_psf(psf_type='gauss')

    sim_data = descwl_shear_sims.make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=coadd_dim,
        bands=bands,
        g1=0,
        g2=0,
        psf=psf,
        se_dim=coadd_dim,
    )
    return sim_data


def detect_and_deblend(mbexp):
    import lsst.afw.table as afw_table
    from lsst.meas.base import (
        SingleFrameMeasurementConfig,
        SingleFrameMeasurementTask,
    )
    from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
    from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig

    detexp = util.coadd_exposures(mbexp.singles)

    schema = afw_table.SourceTable.makeMinimalSchema()

    # note we won't run any of these measurements, but it is needed so that
    # getCentroid will return the peak position rather than NaN.
    # I think it modifies the schema and sets defaults

    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_PsfFlux",
        "base_SkyCoord",
    ]

    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None

    # goes with SdssShape above
    meas_config.slots.shape = None

    # fix odd issue where it thinks things are near the edge
    meas_config.plugins['base_SdssCentroid'].binmax = 1

    _ = SingleFrameMeasurementTask(
        config=meas_config,
        schema=schema,
    )

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = 5
    detection_task = SourceDetectionTask(config=detection_config)

    deblend_task = SourceDeblendTask(
        config=SourceDeblendConfig(),
        schema=schema,
    )

    table = afw_table.SourceTable.make(schema)
    result = detection_task.run(table, detexp)

    if result is not None:
        sources = result.sources
        deblend_task.run(detexp, sources)
    else:
        sources = []

    return sources


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
def test_noise_replacer():
    import lsst.afw.image as afw_image
    seed = 981
    rng = np.random.RandomState(seed)

    sim = make_lsst_sim(rng)

    exps = [texps[0] for _, texps in sim['band_data'].items()]
    mbexp = util.get_mbexp(exps)
    sources = detect_and_deblend(mbexp)

    exposure = mbexp.singles[0]
    exp_copy = afw_image.ExposureF(exposure, deep=True)
    with util.ContextNoiseReplacer(exposure, sources, rng) as replacer:

        assert np.any(exp_copy.image.array != exposure.image.array)

        replaced_copy = afw_image.ExposureF(exposure, deep=True)
        for source in sources[:5]:
            with replacer.sourceInserted(source.getId()):
                assert np.any(exp_copy.image.array != exposure.image.array)

            assert np.all(replaced_copy.image.array == exposure.image.array)

    assert np.all(exp_copy.image.array == exposure.image.array)


@pytest.mark.skipif(
    skip_tests_on_simulations,
    reason='descwl_shear_sims not available'
)
def test_multiband_noise_replacer(show=False):
    import lsst.afw.image as afw_image
    from metadetect.lsst import vis

    seed = 981
    rng = np.random.RandomState(seed)

    sim = make_lsst_sim(rng)

    exps = [texps[0] for _, texps in sim['band_data'].items()]
    mbexp = util.get_mbexp(exps)

    sources = detect_and_deblend(mbexp)

    exp_copys = [afw_image.ExposureF(exp, deep=True) for exp in exps]
    mbexp_copy = util.get_mbexp(exp_copys)

    bands = list(sim['band_data'].keys())

    with util.MultibandNoiseReplacer(mbexp, sources, rng) as replacer:
        if show:
            vis.compare_mbexp(mbexp_copy, replacer.mbexp)

        for exp_copy, exp in zip(exp_copys, replacer.mbexp.singles):
            assert np.any(exp_copy.image.array != exp.image.array)

        replaced_copys = [
            afw_image.ExposureF(exp, deep=True) for exp in replacer.mbexp.singles
        ]
        rmbexp_copy = util.get_mbexp(replaced_copys)
        for source in sources[:5]:
            source_id = source.getId()
            with replacer.sourceInserted(source_id):

                if show:
                    vis.compare_mbexp(rmbexp_copy, replacer.mbexp)

                for band, rexp, exp in zip(
                    bands, replaced_copys, replacer.mbexp.singles
                ):

                    assert np.any(rexp.image.array != exp.image.array), (
                        f'after inserting source {source_id}, band {band} does not '
                        f'differ from replaced exp'
                    )

            for cexp, exp in zip(replaced_copys, replacer.mbexp.singles):
                assert np.all(cexp.image.array == exp.image.array), (
                    f'band {band} did not get restored to noise'
                )

    for cexp, exp in zip(exp_copys, mbexp.singles):
        assert np.all(cexp.image.array == exp.image.array)


if __name__ == '__main__':
    test_multiband_noise_replacer(show=True)
