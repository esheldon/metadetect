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

    # In units of sky noise.  Default for lsst is 5
    'detect_thresh': 5.0,

    # deblending always occurs; this says to do the shear measurements on
    # deblended stamps
    'use_deblended_stamps': False,

    # do sky sub on each coadd input
    'subtract_sky': False,
}

DEFAULT_WEIGHT_FWHMS = {
    'wmom': 1.2,
    'ksigma': 2.0,
}

DEFAULT_STAMP_SIZES = {
    'wmom': 32,
    # TODO determine a good value for this. We used 48 in DES
    # which would be 64 for lsst
    'ksigma': 64,
}
