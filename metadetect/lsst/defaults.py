from copy import deepcopy


DEFAULT_PGAUSS_FWHM = 2.0

DEFAULT_PGAUSS_CONFIG = {
    'fwhm': DEFAULT_PGAUSS_FWHM,
}

DEFAULT_STAMP_SIZE = 49

# threshold for detection
DEFAULT_THRESH = 5.0

# whether to find and subtract the sky, happens before metacal
DEFAULT_SUBTRACT_SKY = False

# Control of the metacal process
# currently we don't have any defaults
DEFAULT_METACAL_CONFIG = {}

# detection config, this may expand
DEFAULT_DETECT_CONFIG = {
    'thresh': DEFAULT_THRESH,
}

# the pgauss subconfig and the stamp_size defaults we be filled in
# programatically based on the measurement_type
DEFAULT_MDET_CONFIG = {
    'subtract_sky': DEFAULT_SUBTRACT_SKY,
    'detect': deepcopy(DEFAULT_DETECT_CONFIG),
    'metacal': deepcopy(DEFAULT_METACAL_CONFIG),
    'pgauss': deepcopy(DEFAULT_PGAUSS_CONFIG),
    'stamp_size': DEFAULT_STAMP_SIZE,
    'shear_bands': None,
}
