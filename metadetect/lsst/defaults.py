from copy import deepcopy


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
DEFAULT_DEBLENDER = 'scarlet'
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

DEFAULT_SHREDDER_CONFIG = {
    'psf_ngauss': 3,
    'init_model': 'exp',
    'miniter': 40,
    'maxiter': 500,
    'flux_miniter': 20,
    'flux_maxiter': 500,
    'tol': 0.001,
}

# the weight subconfig and the stamp_size defaults we be filled in
# programatically based on the measurement_type
DEFAULT_MDET_CONFIG = {
    'meas_type': 'wmom',
    'subtract_sky': DEFAULT_SUBTRACT_SKY,
    'detect': deepcopy(DEFAULT_DETECT_CONFIG),
    'deblend': DEFAULT_DEBLEND,
    'deblender': DEFAULT_DEBLENDER,
    'shredder_config': None,
    'psf': deepcopy(DEFAULT_PSF_CONFIG),
    'metacal': deepcopy(DEFAULT_METACAL_CONFIG),
    'weight': None,
    'stamp_size': None,
}
