import time
import numpy as np
import ngmix
import galsim
import galsim.des
from metadetect import metadetect_and_cal

from fit_des_psf import ShpPSF
from cmcsampler import CMCSampler

PIXSCALE = 0.263

DEFAULT_SIM_CONFIG = {
    'nband': 1,
    'noises': (2.0,),  # this is the noise for 4 coadds stacked togather
    'band_names': ('r',),
    'dims': (225, 225),
    'buff': 25,
    'psf_fwhm': 0.9,
    'g1': 0.02,
    'g2': 0.0,
    'shear_scene': False,
}

TEST_METADETECT_CONFIG = {
    'weight': {
        'fwhm': 1.2,  # arcsec
    },

    'metacal': {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        'use_noise_image': True,
    },

    'sx': {
        # in sky sigma
        # DETECT_THRESH
        'detect_thresh': 0.8,

        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        'deblend_cont': 0.00001,

        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        'minarea': 4,

        'filter_type': 'conv',

        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        'filter_kernel':  [
            [0.004963, 0.021388, 0.051328, 0.068707,
             0.051328, 0.021388, 0.004963],
            [0.021388, 0.092163, 0.221178, 0.296069,
             0.221178, 0.092163, 0.021388],
            [0.051328, 0.221178, 0.530797, 0.710525,
             0.530797, 0.221178, 0.051328],
            [0.068707, 0.296069, 0.710525, 0.951108,
             0.710525, 0.296069, 0.068707],
            [0.051328, 0.221178, 0.530797, 0.710525,
             0.530797, 0.221178, 0.051328],
            [0.021388, 0.092163, 0.221178, 0.296069,
             0.221178, 0.092163, 0.021388],
            [0.004963, 0.021388, 0.051328, 0.068707,
             0.051328, 0.021388, 0.004963],
        ]
    },

    'meds': {
        'min_box_size': 32,
        'max_box_size': 256,

        'box_type': 'iso_radius',

        'rad_min': 4,
        'rad_fac': 2,
        'box_padding': 2,
    },

    # needed for PSF symmetrization
    'psf': {
        'model': 'gauss',

        'ntry': 2,

        'lm_pars': {
            'maxfev': 2000,
            'ftol': 1.0e-5,
            'xtol': 1.0e-5,
        }
    },

    # check for an edge hit
    'bmask_flags': 2**30,
}


class Sim(dict):
    def __init__(self, rng, config=None):
        self.update(DEFAULT_SIM_CONFIG)

        if config is not None:
            self.update(config)

        # only fill the middle part of the image
        self._rng = rng
        self._im_cen = (np.array(self['dims']) - 1) / 2

        self._wcs = galsim.PixelScale(PIXSCALE)
        # stores the PSFEx PSF in world coords
        self._psf = ShpPSF('shp_psf.fit')

        self._cmcsampler = CMCSampler(rng=self._rng)

        frac = 1.0 - self['buff'] * 2 / self['dims'][0]

        self['pos_width'] = self['dims'][0] * frac * 0.5 * PIXSCALE

        assert len(self['band_names']) == self['nband']
        assert len(self['noises']) == self['nband']

        # compute number of objects
        # for the coadd sims, one uses 80000 objects
        # a coadd is 10k * 10k
        # this sim dims[0] * dims[1] but we only use frac * frac of the area
        # so the number of things we want is
        # dims[0] * dims[1] / 1e4^2 * 80000 * frac * frac
        self['nobj'] = int(
            self['dims'][0] * self['dims'][1] / 1e8 * 80000 *
            frac * frac)

    def get_mbobs(self):
        """
        get a simulated MultiBandObsList
        """
        all_band_obj, offsets = self._get_band_objects()

        mbobs = ngmix.MultiBandObsList()
        for band in range(self['nband']):
            band_objects = [o[band] for o in all_band_obj]

            # render objects in loop and sum into final image
            im = galsim.ImageD(
                ncol=self['dims'][1],
                nrow=self['dims'][0],
                wcs=self._wcs)
            for obj, offset in zip(band_objects, offsets):
                obj.drawImage(
                    image=im,
                    offset=offset,
                    add_to_image=True
                )
            im = im.array.copy()

            im += self._rng.normal(scale=self['noises'][band], size=im.shape)
            wt = im*0 + 1.0/self['noises'][band]**2
            bmask = np.zeros(im.shape, dtype='i4')
            noise = self._rng.normal(size=im.shape) / np.sqrt(wt)

            galsim_jac = self._get_loacal_jacobian(
                x=self._im_cen[1], y=self._im_cen[0])

            jac = ngmix.jacobian.Jacobian(
                row=self._im_cen[0],
                col=self._im_cen[1],
                wcs=galsim_jac,
            )

            obs = ngmix.Observation(
                im,
                weight=wt,
                bmask=bmask,
                jacobian=jac,
                psf=self._render_psf(
                    x=self._im_cen[1],
                    y=self._im_cen[0]),
                noise=noise,
            )

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs

    def get_wcs_jac_func(self):
        def _func(row, col):
            galsim_jac = self._get_loacal_jacobian(
                x=col+1, y=row+1)
            return {
                'dudcol': galsim_jac.dudx,
                'dudrow': galsim_jac.dudy,
                'dvdcol': galsim_jac.dvdx,
                'dvdrow': galsim_jac.dvdy}
        return _func

    def _get_eta(self):
        return self._rng.normal(size=2) * 0.4

    def _get_dxdy(self):
        return self._rng.uniform(
            low=-self['pos_width'],
            high=self['pos_width'],
            size=2)

    def _get_band_objects(self):
        all_band_obj = []
        offsets = []

        for i in range(self['nobj']):

            cmcdata = self._cmcsampler.sample()
            band_mags = [
                cmcdata['Mapp_DES_%s' % band] for band in self['band_names']]
            band_fluxes = 10**(0.4 * (30 - np.array(band_mags)))

            eta1, eta2 = self._get_eta()
            dx, dy = self._get_dxdy()

            disk_obj = galsim.Sersic(
                half_light_radius=cmcdata['halflightradius'],
                n=cmcdata['sersicindex']
            ).shear(
                eta1=eta1,
                eta2=eta2)

            # compute the final image position
            if self['shear_scene']:
                shear_mat = galsim.Shear(
                    g1=self['g1'], g2=self['g2']).getMatrix()
                sdx, sdy = np.dot(shear_mat, np.array([dx, dy]) / PIXSCALE)
            else:
                sdx = dx
                sdy = dy

            offset = galsim.PositionD(x=sdx, y=sdy)
            psf_offset = galsim.PositionD(
                x=sdx + self._im_cen[1], y=sdy + self._im_cen[0])
            offsets.append(offset)

            band_objs = []
            for band, flux in enumerate(band_fluxes):
                band_disk = disk_obj.withFlux(flux)
                obj = band_disk.shear(
                    g1=self['g1'], g2=self['g2'])
                obj = galsim.Convolve(obj, self._psf.getPSF(psf_offset))
                band_objs.append(obj)

            all_band_obj.append(band_objs)

        return all_band_obj, offsets

    def _get_loacal_jacobian(self, *, x, y):
        return self._wcs.jacobian(
            image_pos=galsim.PositionD(x=x+1, y=x+1))

    def get_psf_rec_funcs(self):
        funcs = []
        for _ in range(self['nband']):

            def _func(row, col):
                galsim_jac = self._get_loacal_jacobian(x=col, y=row)
                psf_im = self._psf.getPSF(
                    galsim.PositionD(x=col+1, y=row+1)).drawImage(
                        nx=33,
                        ny=33,
                        wcs=galsim_jac).array
                noise = psf_im.max()/1000.0
                psf_im += self._rng.normal(
                    scale=noise,
                    size=psf_im.shape)
                return psf_im

            funcs.append(_func)

        return funcs

    def _render_psf(self, *, x, y):
        galsim_jac = self._get_loacal_jacobian(x=x, y=y)

        psf_im = self._psf.getPSF(
            galsim.PositionD(x=x+1, y=x+1)).drawImage(
                nx=33,
                ny=33,
                wcs=galsim_jac).array

        noise = psf_im.max()/1000.0
        weight = psf_im + 1.0/noise**2
        psf_im += self._rng.normal(
            scale=noise,
            size=psf_im.shape
        )

        cen = (np.array(psf_im.shape)-1.0)/2.0
        j = ngmix.jacobian.Jacobian(row=cen[0], col=cen[1], wcs=galsim_jac)

        psf_obs = ngmix.Observation(
            psf_im,
            weight=weight,
            jacobian=j
        )
        return psf_obs


if __name__ == '__main__':
    ntrial = 1
    rng = np.random.RandomState()

    tm0 = time.time()

    sim = Sim(rng)
    config = {}
    config.update(TEST_METADETECT_CONFIG)

    for trial in range(ntrial):
        print("trial: %d/%d" % (trial+1, ntrial))

        mbobs = sim.get_mbobs()
        metadetect_and_cal.do_metadetect_and_cal(config, mbobs, rng)

    total_time = time.time()-tm0
    print("time per:", total_time/ntrial)
