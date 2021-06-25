from lsst.meas.algorithms import (
    SubtractBackgroundTask,
    SubtractBackgroundConfig,
)

BP_TO_SKIP = [
    "BAD", "EDGE", "DETECTED", "DETECTED_NEGATIVE", "NO_DATA", "BRIGHT",
]


def determine_and_subtract_sky(exp):
    back_config = SubtractBackgroundConfig(
        ignoredPixelMask=BP_TO_SKIP,
    )
    back_task = SubtractBackgroundTask(config=back_config)

    # returns background data, but we are ignoring it for now
    background = back_task.run(exp)
    return background
