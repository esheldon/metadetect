import logging
import warnings

import lsst.afw.table as afw_table
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
)
import lsst.geom as geom
from lsst.pex.exceptions import (
    InvalidParameterError,
    LengthError,
)

from ..procflags import (
    EDGE_HIT, ZERO_WEIGHTS, CENTROID_FAILURE, NO_ATTEMPT,
)
from ..fitting import fit_mbobs_wavg, get_wavg_output_struct

from . import util
from .util import ContextNoiseReplacer
from . import vis
from .defaults import DEFAULT_THRESH

warnings.filterwarnings('ignore', category=FutureWarning)

LOG = logging.getLogger('lsst_measure')


def detect_and_deblend(
    mbexp,
    rng,
    thresh=DEFAULT_THRESH,
    show=False,
):
    """
    run detection and deblending of peaks, as well as basic measurments such as
    centroid.  The SDSS deblender is run in order to split footprints.

    We must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    rng: np.random.RandomState
        Random number generator for noise replacer
    thresh: float, optional
        The detection threshold in units of the sky noise
    show: bool, optional
        If set to True, show images

    Returns
    -------
    sources, detexp
        The sources and the detection exposure
    """
    import lsst.afw.image as afw_image

    if len(mbexp.singles) > 1:
        detexp = util.coadd_exposures(mbexp.singles)
        if detexp is None:
            return [], None
    else:
        detexp = mbexp.singles[0]

    # background measurement within the detection code requires ExposureF
    if not isinstance(detexp, afw_image.ExposureF):
        detexp = afw_image.ExposureF(detexp, deep=True)

    schema = afw_table.SourceTable.makeMinimalSchema()

    # Setup algorithms to run
    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_PsfFlux",
        "base_SkyCoord",
    ]

    # set these slots to none because we aren't running these algorithms
    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None

    # goes with SdssShape above
    meas_config.slots.shape = None

    # fix odd issue where it things things are near the edge
    meas_config.plugins['base_SdssCentroid'].binmax = 1

    meas_task = SingleFrameMeasurementTask(
        config=meas_config,
        schema=schema,
    )

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    # variance here actually means relative to the sqrt(variance)
    # from the variance plane.
    # TODO this would include poisson
    # TODO detection doesn't work right when we tell it to trust
    # the variance
    # detection_config.thresholdType = 'variance'
    detection_config.thresholdValue = thresh

    # these will be ignored when finding the image standard deviation
    detection_config.statsMask = util.get_stats_mask(detexp)

    detection_task = SourceDetectionTask(config=detection_config)

    # these tasks must use the same schema and all be constructed before any
    # other tasks using the same schema are run because schema is modified in
    # place by tasks, and the constructor does a check that fails if we do this
    # afterward

    deblend_task = SourceDeblendTask(
        config=SourceDeblendConfig(),
        schema=schema,
    )

    table = afw_table.SourceTable.make(schema)

    result = detection_task.run(table, detexp)

    if show:
        vis.show_exp(detexp)

    if result is not None:
        sources = result.sources
        deblend_task.run(detexp, sources)

        with ContextNoiseReplacer(detexp, sources, rng) as replacer:

            for source in sources:

                if source.get('deblend_nChild') != 0:
                    continue

                source_id = source.getId()

                with replacer.sourceInserted(source_id):
                    meas_task.callMeasure(source, detexp)

    else:
        sources = []

    return sources, detexp


