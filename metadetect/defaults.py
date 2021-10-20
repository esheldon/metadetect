from copy import deepcopy

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

DEFAULT_WEIGHT_FWHMS = {
    'wmom': 1.2,
    'ksigma': 2.0,
    'pgauss': 2.0,
}

DEFAULT_STAMP_SIZES = {
    'wmom': 32,
    # TODO determine a good value for this. We used 48 in DES
    # which would be 64 for lsst
    'ksigma': 64,
    'am': 64,
    'pgauss': 64,  # TODO would smaller be OK since does not ring?
}

DEFAULT_THRESH = 5.0
DEFAULT_DEBLEND = False
DEFAULT_SUBTRACT_SKY = False
DEFAULT_PSF_CONFIG = {
    'model': 'am',
    'ntry': 4,
}
DEFAULT_METACAL_CONFIG = {
    "use_noise_image": True,
    "psf": "fitgauss",
}
DEFAULT_DETECT_CONFIG = {
    'thresh': DEFAULT_THRESH,
}

# the weight subconfig and the stamp_size defaults we be filled in
# programatically based on the measurement_type
DEFAULT_MDET_CONFIG = {
    'meas_type': 'wmom',
    'subtract_sky': DEFAULT_SUBTRACT_SKY,
    'detect': deepcopy(DEFAULT_DETECT_CONFIG),
    'deblend': DEFAULT_DEBLEND,
    'psf': deepcopy(DEFAULT_PSF_CONFIG),
    'metacal': deepcopy(DEFAULT_METACAL_CONFIG),
    'weight': None,
    'stamp_size': None,
}
