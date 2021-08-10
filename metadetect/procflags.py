NO_ATTEMPT = 2**0
IMAGE_FLAGS = 2**1
PSF_FAILURE = 2**2
OBJ_FAILURE = 2**3
NOMOMENTS_FAILURE = 2**4
BAD_BBOX = 2**5
BMASK_NODET = 2**6

NAME_MAP = {
    # no attempt was made to measure this object, usually
    # due to a previous step in the code fails.  E.g. this
    # will be set for the psf flags if there are IMAGE_FLAGS
    # for the image

    'no_attempt': NO_ATTEMPT,
    NO_ATTEMPT: 'no_attempt',

    # there was an issue with the image data
    'image_flags': IMAGE_FLAGS,
    IMAGE_FLAGS: 'image_flags',

    # psf fitting failed
    PSF_FAILURE: 'psf_failure',
    'psf_failure': PSF_FAILURE,

    # object fitting failed
    OBJ_FAILURE: 'obj_failure',
    'obj_failure': OBJ_FAILURE,

    # moment measurement failed
    NOMOMENTS_FAILURE: 'nomoments_failure',
    'nomoments_failure': NOMOMENTS_FAILURE,

    # there was a problem with the bounding box
    BAD_BBOX: 'bad_bbox',
    'bad_bbox': BAD_BBOX,
}


def get_name(val):
    return NAME_MAP[val]
