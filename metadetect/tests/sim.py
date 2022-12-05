import numpy as np
import ngmix
import galsim

MAX_NBAND = 4
DEFAULT_SIM_CONFIG = {
    'nobj': 4,
    'nband': 3,
    'noises': (0.0005, 0.001, 0.0015, 0.002),
    'scale': 0.263,
    'psf_fwhm': 0.9,
    'dims': (225, 225),
    'flux_low': 0.5,
    'flux_high': 1.5,
    'r50_low': 0.1,
    'r50_high': 2.0,
    'g_std': 0.2,
    'fracdev_low': 0.001,
    'fracdev_high': 0.99,
    'bulge_colors': np.array([0.5, 1.0, 1.5, 2.5]),
    'disk_colors': np.array([1.25, 1.0, 0.75, 0.5]),
}


class Sim(dict):
    def __init__(self, rng, config=None):
        self.update(DEFAULT_SIM_CONFIG)

        if config is not None:
            self.update(config)

        self['pos_width'] = self['dims'][0]/2.0*0.5 * self['scale']
        self.rng = rng

        self._set_wcs()
        self._make_psf()
        self._gpdf = ngmix.priors.GPriorBA(
            self['g_std'],
            rng=self.rng,
        )

    def get_mbobs(self):
        """
        get a simulated MultiBandObsList
        """
        all_band_obj = self._get_band_objects()

        mbobs = ngmix.MultiBandObsList()
        for _band in range(self['nband']):
            band = _band % MAX_NBAND
            band_objects = [o[band] for o in all_band_obj]
            obj = galsim.Sum(band_objects)

            im = obj.drawImage(
                nx=self['dims'][1],
                ny=self['dims'][0],
                scale=self['scale']
            ).array

            im += self.rng.normal(scale=self['noises'][band], size=im.shape)
            wt = im*0 + 1.0/self['noises'][band]**2
            bmask = np.zeros(im.shape, dtype='i4')
            ormask = np.zeros(im.shape, dtype='i4')
            nse = self.rng.normal(scale=self['noises'][band], size=im.shape)

            obs = ngmix.Observation(
                im,
                weight=wt,
                bmask=bmask,
                ormask=ormask,
                jacobian=self._jacobian,
                psf=self._psf_obs.copy(),
                noise=nse,
                ignore_zero_weight=False,
            )

            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs

    def _get_r50(self):
        return self.rng.uniform(
            low=self['r50_low'],
            high=self['r50_high'],
        )

    def _get_flux(self):
        return self.rng.uniform(
            low=self['flux_low'],
            high=self['flux_high'],
        )

    def _get_fracdev(self):
        return self.rng.uniform(
            low=self['fracdev_low'],
            high=self['fracdev_high'],
        )

    def _get_g(self):
        return self._gpdf.sample2d()

    def _get_dxdy(self):
        return self.rng.uniform(
            low=-self['pos_width'],
            high=self['pos_width'],
            size=2,
        )

    def _get_band_objects(self):

        all_band_obj = []
        for i in range(self['nobj']):
            r50 = self._get_r50()
            flux = self._get_flux()
            fracdev = self._get_fracdev()
            dx, dy = self._get_dxdy()

            g1d, g2d = self._get_g()
            g1b = 0.5*g1d
            g2b = 0.5*g2d

            flux_bulge = fracdev*flux
            flux_disk = (1-fracdev)*flux

            bulge_obj = galsim.DeVaucouleurs(
                half_light_radius=r50
            ).shear(g1=g1b, g2=g2b)

            disk_obj = galsim.Exponential(
                half_light_radius=r50
            ).shear(g1=g1d, g2=g2d)

            band_objs = []
            for band in range(self['nband']):
                band_disk = \
                    disk_obj.withFlux(flux_disk*self['disk_colors'][band])
                band_bulge = \
                    bulge_obj.withFlux(flux_bulge*self['bulge_colors'][band])

                obj = galsim.Sum(band_disk, band_bulge).shift(dx=dx, dy=dy)
                obj = galsim.Convolve(obj, self._psf)
                band_objs.append(obj)

            all_band_obj.append(band_objs)

        return all_band_obj

    def _set_wcs(self):
        self._jacobian = ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=self['scale'],
        )

    def _make_psf(self):

        self._psf = galsim.Gaussian(fwhm=self['psf_fwhm'])

        psf_im = self._psf.drawImage(scale=self['scale']).array
        noise = psf_im.max()/1000.0
        weight = psf_im + 1.0/noise**2
        psf_im += self.rng.normal(
            scale=noise,
            size=psf_im.shape
        )

        cen = (np.array(psf_im.shape)-1.0)/2.0
        j = self._jacobian.copy()
        j.set_cen(row=cen[0], col=cen[1])

        psf_obs = ngmix.Observation(
            psf_im,
            weight=weight,
            jacobian=j
        )
        self._psf_obs = psf_obs


def make_mbobs_sim(
    seed, nband, simulate_star=False, noise_scale=1, band_flux_factors=None,
    band_image_sizes=None, wcs_var_scale=1,
):
    rng = np.random.RandomState(seed=seed)

    if simulate_star:
        gal = galsim.Gaussian(
            fwhm=1e-6,
        ).shear(
            g1=rng.uniform(low=-0.1, high=0.1),
            g2=rng.uniform(low=-0.1, high=0.1),
        ).withFlux(
            400
        )
    else:
        gal = galsim.Exponential(
            half_light_radius=rng.uniform(low=0.5, high=0.7),
        ).shear(
            g1=rng.uniform(low=-0.1, high=0.1),
            g2=rng.uniform(low=-0.1, high=0.1),
        ).withFlux(
            400
        )
    mbobs = ngmix.MultiBandObsList()

    for band in range(nband):
        if band_image_sizes is not None:
            dim = band_image_sizes[band]
        else:
            dim = 35
        cen = (dim-1)/2
        psf_dim = dim + 17
        psf_cen = (psf_dim-1)/2

        psf = galsim.Gaussian(
            fwhm=rng.uniform(low=0.8, high=0.9),
        ).shear(
            g1=rng.uniform(low=-0.1, high=0.1),
            g2=rng.uniform(low=-0.1, high=0.1),
        )

        gs_wcs = galsim.ShearWCS(
            0.25,
            galsim.Shear(
                g1=rng.uniform(low=-0.1, high=0.1) * wcs_var_scale,
                g2=rng.uniform(low=-0.1, high=0.1) * wcs_var_scale,
            )
        ).jacobian()
        offset = rng.uniform(low=-0.5, high=0.5, size=2) * wcs_var_scale

        if band_flux_factors is not None:
            flux_factor = band_flux_factors[band]
        else:
            flux_factor = 1.0
        obj = galsim.Convolve([gal * flux_factor, psf])

        im = obj.drawImage(nx=dim, ny=dim, wcs=gs_wcs, offset=offset).array
        nse = np.sqrt(np.sum(im**2)) / rng.uniform(low=10, high=100) * noise_scale
        im += rng.normal(size=im.shape, scale=nse)

        psf_im = psf.drawImage(nx=psf_dim, ny=psf_dim, wcs=gs_wcs).array
        psf_obs = ngmix.Observation(
            image=psf_im,
            jacobian=ngmix.Jacobian(
                row=psf_cen,
                col=psf_cen,
                wcs=gs_wcs,
            )
        )

        obslist = ngmix.ObsList()
        obslist.append(
            ngmix.Observation(
                image=im,
                weight=np.ones_like(im) / nse**2,
                jacobian=ngmix.Jacobian(
                    row=cen+offset[1],
                    col=cen+offset[0],
                    wcs=gs_wcs,
                ),
                bmask=np.zeros_like(im, dtype=np.int32),
                ormask=(rng.uniform(size=im.shape) * 2**29).astype(int),
                mfrac=rng.uniform(size=im.shape),
                psf=psf_obs,
                meta={"wgt": 1.0/nse**2},
                noise=rng.normal(size=im.shape, scale=nse),
            )
        )
        mbobs.append(obslist)

    return mbobs
