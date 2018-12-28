import time
import numpy as np
import ngmix
import galsim
import galsim.des
from metadetect import metadetect_and_cal

WCS_HEADER = {
    'NAXIS': 2,
    'NAXIS1': 10000,
    'NAXIS2': 10000,
    'CTYPE1': 'RA---TAN',
    'CTYPE2': 'DEC--TAN',
    'CRPIX1': 5000.5,
    'CRPIX2': 5000.5,
    'CD1_1': -7.305555555556E-05,
    'CD1_2': 0.0,
    'CD2_1': 0.0,
    'CD2_2': 7.305555555556E-05,
    'CUNIT1': 'deg     ',
    'CUNIT2': 'deg     ',
    'CRVAL1': 86.176841,
    'CRVAL2': -22.827778,
}

DEFAULT_SIM_CONFIG = {
    'nobj': 4,
    'nband': 3,
    'noises': (0.0005, 0.001, 0.0015),
    'psf_fwhm': 0.9,
    'dims': (225, 225),
    'flux_low': 5,
    'flux_high': 5,
    'r50_low': 2,
    'r50_high': 2,
    'g_std': 0.2,
    'fracdev_low': 0.000,
    'fracdev_high': 0.000,
    'bulge_colors': np.array([0.5, 1.0, 1.5]),
    'disk_colors': np.array([1.25, 1.0, 0.75]),
    'g1': 0.02,
    'g2': 0.0,
    'shape_scale': 0.0,
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
        self['pos_width'] = self['dims'][0]/2.0 * 0.5
        self._rng = rng
        self._im_cen = (np.array(self['dims']) - 1) / 2

        self._wcs = galsim.FitsWCS(header=WCS_HEADER)
        psf_im = galsim.Gaussian(fwhm=self['psf_fwhm']).drawImage(
            nx=33,
            ny=33,
            scale=0.263).array
        self._psf = galsim.InterpolatedImage(
            galsim.Image(psf_im, dtype=psf_im.dtype, copy=True),
            x_interpolant='lanczos3',
            scale=0.263)

        self._gpdf = ngmix.priors.GPriorBA(
            self['g_std'],
            rng=self._rng,
        )

    def get_mbobs(self):
        """
        get a simulated MultiBandObsList
        """
        all_band_obj, obj_im_pos = self._get_band_objects()

        mbobs = ngmix.MultiBandObsList()
        for band in range(self['nband']):
            band_objects = [o[band] for o in all_band_obj]

            im = galsim.ImageD(
                nrow=self['dims'][0],
                ncol=self['dims'][1],
                wcs=self._wcs)
            for obj, (x, y) in zip(band_objects, obj_im_pos):
                dx = x - self._im_cen[0]
                dy = y - self._im_cen[1]
                obj.drawImage(
                    image=im,
                    offset=galsim.PositionD(x=dx, y=dy),
                    add_to_image=True,
                    method='no_pixel')
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

    def _get_r50(self):
        return self._rng.uniform(
            low=self['r50_low'],
            high=self['r50_high'],
        )

    def _get_flux(self):
        return self._rng.uniform(
            low=self['flux_low'],
            high=self['flux_high'],
        )

    def _get_fracdev(self):
        return self._rng.uniform(
            low=self['fracdev_low'],
            high=self['fracdev_high'],
        )

    def _get_g(self):
        return np.array(self._gpdf.sample2d()) * self['shape_scale']

    def _get_dxdy(self):
        return self._rng.uniform(
            low=-self['pos_width'],
            high=self['pos_width'],
            size=2,
        ) + self._im_cen

    def _get_band_objects(self):
        all_band_obj = []
        im_pos = []
        for i in range(self['nobj']):
            r50 = self._get_r50()
            flux = self._get_flux()
            fracdev = self._get_fracdev()

            g1d, g2d = self._get_g()
            g1b = 0.5*g1d
            g2b = 0.5*g2d
            pos = self._get_dxdy()

            flux_bulge = fracdev*flux
            flux_disk = (1-fracdev)*flux
            # print("fracdev:", fracdev)

            bulge_obj = galsim.DeVaucouleurs(
                half_light_radius=r50
            ).shear(
                g1=g1b, g2=g2b
            ).shear(
                g1=self['g1'], g2=self['g2']
            )

            disk_obj = galsim.Exponential(
                half_light_radius=r50
            ).shear(
                g1=g1d, g2=g2d
            ).shear(
                g1=self['g1'], g2=self['g2']
            )

            band_objs = []
            for band in range(self['nband']):
                band_disk = disk_obj.withFlux(
                    flux_disk*self['disk_colors'][band])
                band_bulge = bulge_obj.withFlux(
                    flux_bulge*self['bulge_colors'][band])

                obj = galsim.Sum(band_disk, band_bulge)
                obj = galsim.Convolve(obj, self._psf)
                band_objs.append(obj)

            all_band_obj.append(band_objs)
            im_pos.append(pos)

        return all_band_obj, im_pos

    def _get_loacal_jacobian(self, *, x, y):
        return self._wcs.jacobian(
            image_pos=galsim.PositionD(x=x+1, y=x+1))

    def get_psf_rec_funcs(self):
        funcs = []
        for _ in range(self['nband']):

            def _func(row, col):
                galsim_jac = self._get_loacal_jacobian(x=col, y=row)
                psf_im = self._psf.drawImage(
                    nx=33,
                    ny=33,
                    wcs=galsim_jac,
                    method='no_pixel').array

                noise = psf_im.max()/1000.0
                psf_im += self._rng.normal(
                    scale=noise,
                    size=psf_im.shape
                )

                return psf_im

            funcs.append(_func)

        return funcs

    def _render_psf(self, *, x, y):
        galsim_jac = self._get_loacal_jacobian(x=x, y=y)

        psf_im = self._psf.drawImage(
                nx=33,
                ny=33,
                wcs=galsim_jac,
                method='no_pixel').array

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


def _show_mbobs(mer):
    import images

    mbobs = mer.mbobs

    rgb = images.get_color_image(
        mbobs[2][0].image.transpose(),
        mbobs[1][0].image.transpose(),
        mbobs[0][0].image.transpose(),
        nonlinear=0.1,
    )
    rgb *= 1.0/rgb.max()

    images.view_mosaic(
        [rgb,
         mer.seg,
         mer.detim],
        titles=['image', 'seg', 'detim'],
    )


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
