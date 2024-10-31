import lsst.afw.table as afw_table
from lsst.meas.algorithms import (
    SubtractBackgroundTask,
    SubtractBackgroundConfig,
    SourceDetectionTask,
)
from lsst.meas.algorithms import SourceDetectionConfig as OriginalSourceDetectionConfig
from lsst.pex.config import Config, ConfigurableField, Field
from lsst.pex.exceptions import RuntimeError as LSSTRuntimeError
from lsst.pipe.base import Task, TaskError

from .defaults import DEFAULT_THRESH
from . import util


class SourceDetectionConfig(OriginalSourceDetectionConfig):
    @property
    def thresholdValue(self):
        return self.thresholdValue

    @thresholdValue.setter
    def thresholdValue(self, value):
        self.thresholdValue = value


SourceDetectionTask.ConfigClass = SourceDetectionConfig


class IterateDetectionSkySubConfig(Config):
    niter = Field[int](
        doc="Number of iterations",
        default=2,
    )

    detect = ConfigurableField[SourceDetectionConfig](
        doc="Detection config",
        target=SourceDetectionTask
    )

    back = ConfigurableField[SubtractBackgroundConfig](
        doc="Subtract background config",
        target=SubtractBackgroundTask
    )

    def setDefaults(self):
        super().setDefaults()

        # detection
        self.detect.reEstimateBackground = False
        self.detect.thresholdValue = DEFAULT_THRESH

        # background subtraction
        bp_to_skip = util.get_stats_mask()
        self.back = SubtractBackgroundConfig(ignoredPixelMask=bp_to_skip)


class IterateDetectionSkySubTask(Task):
    ConfigClass = IterateDetectionSkySubConfig
    _DefaultName = "iterate_detection_and_skysub"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("detect")
        self.makeSubtask("back")

    def run(self, exposure):
        if self.config.niter < 1:
            raise ValueError(f'niter {self.config.niter} is less than 1')

        schema = afw_table.SourceTable.makeMinimalSchema()
        table = afw_table.SourceTable.make(schema)

        # keep a running sum of each sky that was subtracted
        try:
            sky_meas = 0.0
            for i in range(self.config.niter):
                self.back.run(exposure)
                result = self.detect.run(table, exposure)

                sky_meas += exposure.getMetadata()['BGMEAN']

            meta = exposure.getMetadata()

            # this is the overall sky we subtracted in all iterations
            meta['BGMEAN'] = sky_meas
        except LSSTRuntimeError as err:
            err = str(err).replace('lsst::pex::exceptions::RuntimeError:', '')
            self.detect.log.warn(err)
            result = None
        except TaskError as err:
            err = str(err).replace('lsst.pipe.base.task.TaskError:', '')
            self.detect.log.warn(err)
            result = None

        return result


class SubtractSkyMbExpConfig(Config):
    iterate_detection_and_skysub = ConfigurableField[IterateDetectionSkySubConfig](
        doc="Iterate detection and sky subtraction config",
        target=IterateDetectionSkySubTask
    )

    def setDefaults(self):
        super().setDefaults()

        self.iterate_detection_and_skysub.detect.thresholdValue = DEFAULT_THRESH
        self.iterate_detection_and_skysub.niter = 2


class SubtractSkyMbExpTask(Task):
    ConfigClass = SubtractSkyMbExpConfig
    _DefaultName = "subtract_sky_mbexp"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("iterate_detection_and_skysub")

    def run(self, mbexp):
        for exp in mbexp:
            self.iterate_detection_and_skysub.run(exposure=exp)


def determine_and_subtract_sky(exp):
    """
    Determine and subtract the sky from the input exposure.
    The exposure is modified.

    Parameters
    ----------
    exp: Exposure
        The exposure to be processed
    """

    bp_to_skip = util.get_stats_mask(exp)
    back_config = SubtractBackgroundConfig(ignoredPixelMask=bp_to_skip)
    back_task = SubtractBackgroundTask(config=back_config)

    # returns background data, but we are ignoring it for now
    background = back_task.run(exp)
    return background


def subtract_sky_mbexp(mbexp, thresh=DEFAULT_THRESH, config=None):
    """
    subtract sky

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    thresh: float
        Threshold for detection
    """
    config_override = config if config is not None else {}

    if thresh:
        if 'iterate_detection_and_skysub' not in config_override:
            config_override['iterate_detection_and_skysub'] = {}
        if 'detect' not in config_override['iterate_detection_and_skysub']:
            config_override['iterate_detection_and_skysub']['detect'] = {}

        detect_override = config_override['iterate_detection_and_skysub']['detect']
        detect_override['thresholdValue'] = thresh

    config = SubtractSkyMbExpConfig()
    config.setDefaults()

    util.override_config(config, config_override)

    config.freeze()
    config.validate()
    task = SubtractSkyMbExpTask(config=config)
    task.run(mbexp=mbexp)


def iterate_detection_and_skysub(
    exposure, thresh, niter=2, config=None
):
    """
    Iterate detection and sky subtraction

    Parameters
    ----------
    exposure: Exposure
        The exposure to process
    thresh: float
        threshold for detection
    niter: int, optional
        Number of iterations for detection and sky subtraction.
        Must be >= 1. Default is 2 which is recommended.

    Returns
    -------
    Result from running the detection task
    """
    if niter < 1:
        raise ValueError(f'niter {niter} is less than 1')

    config_override = config if config is not None else {}

    # set threshold
    if 'detect' not in config_override:
        config_override['detect'] = {}
    config_override['detect']['thresholdValue'] = thresh

    if niter:
        config_override['niter'] = niter

    config = IterateDetectionSkySubConfig()
    config.setDefaults()

    util.override_config(config, config_override)

    task = IterateDetectionSkySubTask(config=config)
    result = task.run(exposure)
    return result


class SkySubtractionTask(Task):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("subtract_background")
        self.makeSubtask("detect_sources")

    def run(self, exposure, **kwargs):
        return self._sky_sub(exposure, **kwargs)
