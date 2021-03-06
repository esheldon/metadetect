import logging
import numpy as np

logger = logging.getLogger(__name__)


class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front == '':
            front = None
        if back == '' or back == 'noshear':
            back = None

        self.front = front
        self.back = back

        if self.front is None and self.back is None:
            self.nomod = True
        else:
            self.nomod = False

    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s_%s' % (self.front, n)
            if self.back is not None:
                n = '%s_%s' % (n, self.back)

        return n


def trim_odd_image(im):
    """
    trim an odd dimension image to by square and with equal distance from
    canonical center to all edges
    """

    dims = im.shape
    if dims[0] != dims[1]:
        logger.debug('original dims: %s' % str(dims))
        assert dims[0] % 2 != 0, 'image must have odd dims'
        assert dims[1] % 2 != 0, 'image must have odd dims'

        dims = np.array(dims)
        cen = (dims-1)//2
        cen = cen.astype('i4')

        distances = (
            cen[0]-0,
            dims[0]-cen[0]-1,
            cen[1]-0,
            dims[1]-cen[1]-1,
        )
        logger.debug('distances: %s' % str(distances))
        min_dist = min(distances)

        start_row = cen[0] - min_dist
        end_row = cen[0] + min_dist
        start_col = cen[1] - min_dist
        end_col = cen[1] + min_dist

        # adding +1 for slices
        new_im = im[
            start_row:end_row+1,
            start_col:end_col+1,
        ].copy()

        logger.debug('new dims: %s' % str(new_im.shape))

    else:
        new_im = im

    return new_im


def get_ored_bits(maskobj, bitnames):
    """
    get or of bits

    Parameters
    ----------
    maskobj: lsst mask obj
        Must have method getPlaneBitMask
    bitnames: list of strings
        list of bitmask names
    """
    bits = 0
    for ibit, bitname in enumerate(bitnames):
        bitval = maskobj.getPlaneBitMask(bitname)
        bits |= bitval

    return bits


try:
    # only in ngmix v2 - can remove eventually
    from ngmix.util import get_ratio_var, get_ratio_error
except ImportError:
    def get_ratio_var(a, b, var_a, var_b, cov_ab):
        """
        get variance in a/b
        """

        if b == 0:
            raise ValueError("zero in denominator")

        rsq = (a/b)**2

        var = rsq * (var_a/a**2 + var_b/b**2 - 2*cov_ab/(a*b))
        return var

    def get_ratio_error(a, b, var_a, var_b, cov_ab):
        """
        get error on a/b
        """
        from math import sqrt

        var = get_ratio_var(a, b, var_a, var_b, cov_ab)

        if var < 0:
            var = 0
        error = sqrt(var)
        return error
