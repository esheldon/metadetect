import copy

import ngmix.flags
from ngmix.flags import NO_ATTEMPT  # noqa

# these flags start at 19 always
# this allows us to combine them with the flags in ngmix
SHEAR_RANGE_ERROR = 2**19
EDGE_HIT = 2**20
PSF_FAILURE = 2**21
OBJ_FAILURE = 2**22
NOMOMENTS_FAILURE = 2**23
BAD_BBOX = 2**24
ZERO_WEIGHTS = 2**25
NO_DATA = 2**26
MISSING_BAND = 2**27
INCONSISTENT_BANDS = 2**28
CENTROID_FAILURE = 2**29
DEBLEND_FAILURE = 2**30
# DO NOT use 2**31 or higher

NAME_MAP = copy.deepcopy(ngmix.flags.NAME_MAP)
NAME_MAP[SHEAR_RANGE_ERROR] = "shear is out of range (abs(g) > 1 or abs(g_1,2) > 1)"
NAME_MAP[EDGE_HIT] = "bbox hit edge"
NAME_MAP[PSF_FAILURE] = "PSF fit failed"
NAME_MAP[OBJ_FAILURE] = "object fit failed"
NAME_MAP[NOMOMENTS_FAILURE] = "no moments"
NAME_MAP[BAD_BBOX] = "problem making bounding box"
NAME_MAP[ZERO_WEIGHTS] = "weights all zero"
NAME_MAP[NO_DATA] = "no data"
NAME_MAP[MISSING_BAND] = "missing data in one or more bands"
NAME_MAP[INCONSISTENT_BANDS] = (
    "# of bands for PSF vs shear vs flux is not correct or "
    "there was a stamp coadding issue"
)
NAME_MAP[CENTROID_FAILURE] = "finding the centroid failed"
NAME_MAP[DEBLEND_FAILURE] = "The deblending failed"

for k, v in list(NAME_MAP.items()):
    NAME_MAP[v] = k


def get_procflags_str(val):
    """Get a descriptive string given a flag value.

    Parameters
    ----------
    val : int
        The flag value.

    Returns
    -------
    flagstr : str
        A string of descriptions for each bit separated by `|`.
    """
    return ngmix.flags.get_flags_str(val, name_map=NAME_MAP)
