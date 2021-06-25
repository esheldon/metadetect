from lsst.meas.algorithms import (
    SubtractBackgroundTask,
    SubtractBackgroundConfig,
)

BP_TO_SKIP = [
    "BAD", "EDGE", "DETECTED", "DETECTED_NEGATIVE", "NO_DATA", "BRIGHT",
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
    back_config = SubtractBackgroundConfig(
        ignoredPixelMask=BP_TO_SKIP,
    )
    back_task = SubtractBackgroundTask(config=back_config)

    # returns background data, but we are ignoring it for now
    background = back_task.run(exp)
    return background
