DEFAULT_LOGLEVEL = 'INFO'
BMASK_EDGE = 2**30
DEFAULT_IMAGE_VALUES = {
    'image': 0.0,
    'weight': 0.0,
    'seg': 0,
    'bmask': BMASK_EDGE,
    'noise': 0.0,
}

ALLOWED_BOX_SIZES = [
    2,3,4,6,8,12,16,24,32,48,  # noqa
    64,96,128,192,256,  # noqa
    384,512,768,1024,1536,  # noqa
    2048,3072,4096,6144  # noqa
]

# stamp size is not a default, depends on measurement
DEFAULT_MDET_CONFIG = {
    # wmom or ksigma
    'meas_type': 'wmom',

    # fitgauss or gauss
    'metacal_psf': 'fitgauss',

    # 1.2 prob. good for wmom, 2.0 for ksigma but need to test
    'weight_fwhm': 1.2,

    # default for lsst is 5
    'detect_thresh': 5.0,

    # deblending always occurs, this says to do measurements on deblended
    # stamps
    'use_deblended_stamps': False,

    # do sky sub on each coadd input
    'subtract_sky': False,
}

