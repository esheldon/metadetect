import numpy
import fitsio


class CMCSampler(object):
    def __init__(
            self,
            file_name='matched_cosmos_cat_25.2.fits',
            sersic_min=0.3, sersic_max=5.,
            hlr_min=0.1, hlr_max=1.5,
            mag_i_max=30., rng=None, replace=True):

        # Read in data
        cmc_data = fitsio.read(file_name)
        use = (
            (cmc_data["sersicindex"] > sersic_min) *
            (cmc_data["sersicindex"] < sersic_max) *
            (cmc_data["halflightradius"] > hlr_min) *
            (cmc_data["halflightradius"] < hlr_max) *
            (cmc_data["Mapp_DES_i"] < mag_i_max))
        self.cmc_data = cmc_data[use]
        self.colnames = self.cmc_data.dtype.names
        self.replace = replace
        if rng is None:
            self.rng = numpy.random.RandomState(seed=0)
        else:
            self.rng = rng

    def sample(self):
        index = self.rng.randint(0, len(self.cmc_data))
        to_retutrn = {
            c: v for c, v in zip(self.colnames, self.cmc_data[index])}
        if self.replace is False:
            self.cmc_data = numpy.delete(self.cmc_data, index)
        return to_retutrn
