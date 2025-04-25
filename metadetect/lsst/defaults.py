from copy import deepcopy


DEFAULT_WEIGHT_FWHMS = {
    'wmom': 1.2,
    'ksigma': 2.0,
    'pgauss': 2.0,
}
DEFAULT_FWHM_SMOOTH = 0
DEFAULT_FWHM_REG = 0

DEFAULT_STAMP_SIZES = {
    'wmom': 32,
    'ksigma': 64,
    'pgauss': 49,
    'am': 49,
    'gauss': 49,
}

# threshold for detection
DEFAULT_THRESH = 5.0

# whether to find and subtract the sky, happens before metacal
DEFAULT_SUBTRACT_SKY = False

# config for fitting the original psfs
DEFAULT_PSF_CONFIG = {
    'model': 'am',
    'ntry': 4,
}

# Control of the metacal process
# not currently used for new metacal_exposures code that always
DEFAULT_METACAL_CONFIG = {
    "use_noise_image": True,
    "psf": "fitgauss",
}

# detection config, this may expand
DEFAULT_DETECT_CONFIG = {
    'thresh': DEFAULT_THRESH,
}

# the weight subconfig and the stamp_size defaults we be filled in
# programatically based on the measurement_type
DEFAULT_MDET_CONFIG = {
    'meas_type': 'wmom',
    'subtract_sky': DEFAULT_SUBTRACT_SKY,
    'detect': deepcopy(DEFAULT_DETECT_CONFIG),
    'psf': deepcopy(DEFAULT_PSF_CONFIG),
    'metacal': deepcopy(DEFAULT_METACAL_CONFIG),
    'weight': None,
    'stamp_size': None,
}
