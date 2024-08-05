from lsst.meas.algorithms import SourceDetectionConfig


def test_detect_config():
    expected_keys = [
        'minPixels',
        'isotropicGrow',
        'combinedGrow',
        'nSigmaToGrow',
        'returnOriginalFootprints',
        'thresholdValue',
        'includeThresholdMultiplier',
        'thresholdType',
        'thresholdPolarity',
        'adjustBackground',
        'reEstimateBackground',
        'background',
        'tempLocalBackground',
        'doTempLocalBackground',
        'tempWideBackground',
        'doTempWideBackground',
        'nPeaksMaxSimple',
        'nSigmaForKernel',
        'statsMask',
        'excludeMaskPlanes'
    ]

    config = SourceDetectionConfig()

    keys = config.toDict().keys()

    for key in keys:
        assert key in expected_keys, f'found unexpected config key {key}'
