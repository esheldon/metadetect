DEFAULT_METACAL_PARS = {
    'symmetrize_psf': True,
    'types': ['noshear','1p','1m','2p','2m'],
}

DEFAULT_SX_CONFIG = {
    # in sky sigma
    #DETECT_THRESH
    'detect_thresh': 0.8,

    # Minimum contrast parameter for deblending
    #DEBLEND_MINCONT
    'deblend_cont': 0.00001,

    # minimum number of pixels above threshold
    #DETECT_MINAREA: 6
    'minarea': 4,

    'filter_type': 'conv',

    # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
    'filter_kernel':  [
        [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
        [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
        [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
        [0.068707, 0.296069, 0.710525, 0.951108, 0.710525, 0.296069, 0.068707],
        [0.051328, 0.221178, 0.530797, 0.710525, 0.530797, 0.221178, 0.051328],
        [0.021388, 0.092163, 0.221178, 0.296069, 0.221178, 0.092163, 0.021388],
        [0.004963, 0.021388, 0.051328, 0.068707, 0.051328, 0.021388, 0.004963],
    ]

}

DEFAULT_MEDS_CONFIG = {
    'min_box_size': 32,
    'max_box_size': 256,

    'box_type': 'iso_radius',

    'rad_min': 4,
    'rad_fac': 2,
    'box_padding': 2,
}

BMASK_EDGE=2**30
DEFAULT_IMAGE_VALUES = {
    'image':0.0,
    'weight':0.0,
    'seg':0,
    'bmask':BMASK_EDGE,
}

ALLOWED_BOX_SIZES = [
    2,3,4,6,8,12,16,24,32,48,
    64,96,128,192,256,
    384,512,768,1024,1536,
    2048,3072,4096,6144
]


