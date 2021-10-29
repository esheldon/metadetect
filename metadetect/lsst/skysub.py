import lsst.afw.table as afw_table
from lsst.meas.algorithms import (
    SubtractBackgroundTask,
    SubtractBackgroundConfig,
    SourceDetectionTask,
    SourceDetectionConfig,
)

from .defaults import DEFAULT_THRESH

BP_TO_SKIP = [
    "BAD", "EDGE", "DETECTED", "DETECTED_NEGATIVE", "NO_DATA",
]


def determine_and_subtract_sky(exp):
    """
    Determine and subtract the sky from the input exposure.
    The exposure is modified.

    Parameters
    ----------
    exp: Exposure
        The exposure to be processed
    """
    if 'BRIGHT' in exp.mask.getMaskPlaneDict():
        bp_to_skip = BP_TO_SKIP + ['BRIGHT']
    else:
        bp_to_skip = BP_TO_SKIP

    back_config = SubtractBackgroundConfig(
        ignoredPixelMask=bp_to_skip,
    )
    back_task = SubtractBackgroundTask(config=back_config)

    # returns background data, but we are ignoring it for now
    background = back_task.run(exp)
    return background


def subtract_sky_mbobs(mbobs, thresh=DEFAULT_THRESH):
    """
    subtract sky

    Parameters
    ----------
    mbobs: ngmix.MultiBandObsList
        The observations to sky subtract
    thresh: float
        Threshold for detection
    """
    for obslist in mbobs:
        for obs in obslist:
            exp = obs.coadd_exp

            iterate_detection_and_skysub(
                exposure=exp,
                thresh=thresh,
            )

            obs.image = exp.image.array


def iterate_detection_and_skysub(
    exposure, thresh, niter=2,
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
    from lsst.pex.exceptions import RuntimeError as LSSTRuntimeError
    from lsst.pipe.base.task import TaskError
    if niter < 1:
        raise ValueError(f'niter {niter} is less than 1')

    schema = afw_table.SourceTable.makeMinimalSchema()
    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    detection_config.thresholdValue = thresh
    detection_task = SourceDetectionTask(config=detection_config)

    table = afw_table.SourceTable.make(schema)

    # keep a running sum of each sky that was subtracted
    try:
        sky_meas = 0.0
        for i in range(niter):
            determine_and_subtract_sky(exposure)
            result = detection_task.run(table, exposure)

            sky_meas += exposure.getMetadata()['BGMEAN']

        meta = exposure.getMetadata()

        # this is the overall sky we subtracted in all iterations
        meta['BGMEAN'] = sky_meas
    except LSSTRuntimeError as err:
        err = str(err).replace('lsst::pex::exceptions::RuntimeError:', '')
        detection_task.log.warn(err)
        result = None
    except TaskError as err:
        err = str(err).replace('lsst.pipe.base.task.TaskError:', '')
        detection_task.log.warn(err)
        result = None

    return result
