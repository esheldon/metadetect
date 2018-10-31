"""

Code to take a set of images, for example coadds in different bands, run
detection on them and produce a MEDS like interface to get postage stamps

"""
from __future__ import print_function
import logging
import numpy as np
from numpy import pi
import esutil as eu
from esutil.numpy_util import between
import meds
import ngmix
import time

from . import moflib

logger = logging.getLogger(__name__)

FWHM_FAC = 2*np.sqrt(2*np.log(2))

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

class MultiBandMEDS(object):
    """
    Interface to MEDS objects in more than one band.
    """
    def __init__(self, mlist):
        self.mlist=mlist

    def get_mbobs_list(self, indices=None, weight_type='weight'):
        """
        get a list of MultiBandObsList for every object or
        the specified indices
        """

        if indices is None:
            indices = np.arange(self.mlist[0].size)

        list_of_obs=[]
        for iobj in indices:
            mbobs=self.get_mbobs(iobj, weight_type=weight_type)
            list_of_obs.append(mbobs)

        return list_of_obs

    def get_mbobs(self, iobj, weight_type='weight'):
        """
        get a multiband obs list
        """
        mbobs=ngmix.MultiBandObsList()

        for m in self.mlist:
            obslist = m.get_obslist(iobj, weight_type=weight_type)
            mbobs.append(obslist)

        return mbobs

class MEDSInterface(meds.MEDS):
    """
    Wrap a full image with a MEDS-like interface
    """
    def __init__(self, image, weight, seg, bmask, cat):
        self._image_data=dict(
            image=image,
            weight=weight,
            seg=seg,
            bmask=bmask,
        )
        self._cat=cat

    def get_cutout(self, iobj, icutout, type='image'):
        """
        Get a single cutout for the indicated entry

        parameters
        ----------
        iobj:
            Index of the object
        icutout:
            Index of the cutout for this object.
        type: string, optional
            Cutout type. Default is 'image'.  Allowed
            values are 'image','weight','seg','bmask'

        returns
        -------
        The cutout image
        """

        if type=='psf':
            return self.get_psf(iobj,icutout)

        self._check_indices(iobj, icutout=icutout)

        if type not in self._image_data:
            raise ValueError("bad cutout type: '%s'" % type)

        im=self._image_data[type]
        dims = im.shape

        c=self._cat
        orow=c['orig_start_row'][iobj,icutout]
        ocol=c['orig_start_col'][iobj,icutout]
        erow=c['orig_end_row'][iobj,icutout]
        ecol=c['orig_end_col'][iobj,icutout]
        bsize=c['box_size'][iobj]

        orow_box, row_box = self._get_clipped_boxes(dims[0],orow,bsize)
        ocol_box, col_box = self._get_clipped_boxes(dims[1],ocol,bsize)

        read_im = im[orow_box[0]:orow_box[1],
                     ocol_box[0]:ocol_box[1]]

        subim = np.zeros( (bsize, bsize), dtype=im.dtype)
        subim += DEFAULT_IMAGE_VALUES[type]

        subim[row_box[0]:row_box[1],
              col_box[0]:col_box[1]] = read_im

        return subim

    def _get_clipped_boxes(self, dim, start, bsize):
        """
        get clipped boxes for slicing

        If the box size goes outside the dimensions,
        trim them back

        parameters
        ----------
        dim: int
            Dimension of this axis
        start: int
            Starting position in the image for this axis
        bsize: int
            Size of box

        returns
        -------
        obox, box

        obox: [start,end]
            Start and end slice ranges in the original image
        box: [start,end]
            Start and end slice ranges in the output image
        """
        # slice range in the original image
        obox = [start, start+bsize]

        # slice range in the sub image into which we will copy
        box = [0, bsize]

        # rows
        if obox[0] < 0:
            obox[0] = 0
            box[0] = 0 - start

        im_max = dim
        diff= im_max - obox[1]
        if diff < 0:
            obox[1] = im_max
            box[1] = box[1] + diff

        return obox, box


    def get_obslist(self, iobj, weight_type='weight'):
        """
        get an ngmix ObsList for all observations

        parameters
        ----------
        iobj:
            Index of the object
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
            Default is 'weight'

        returns
        -------
        an ngmix ObsList
        """

        import ngmix
        obslist=ngmix.ObsList()
        for icut in xrange(self._cat['ncutout'][iobj]):
            obs=self.get_obs(iobj, icut, weight_type=weight_type)
            obslist.append(obs)

        obslist.meta['flux'] = obs.meta['flux']
        obslist.meta['T'] = obs.meta['T']
        return obslist

    def get_obs(self, iobj, icutout, weight_type='weight'):
        """
        get an ngmix Observation

        parameters
        ----------
        iobj:
            Index of the object
        icutout:
            Index of the cutout for this object.
        weight_type: string, optional
            Weight type. can be one of
                'weight': the actual weight map
                'uberseg': uberseg modified weight map
            Default is 'weight'

        returns
        -------
        an ngmix Observation
        """

        import ngmix
        im=self.get_cutout(iobj, icutout, type='image')
        bmask=self.get_cutout(iobj, icutout, type='bmask')
        jd=self.get_jacobian(iobj, icutout)


        if weight_type=='uberseg':
            wt=self.get_uberseg(iobj, icutout)
        elif weight_type=='cweight':
            wt=self.get_cweight_cutout(iobj, icutout, restrict_to_seg=True)
        elif weight_type=='weight':
            wt=self.get_cutout(iobj, icutout, type='weight')
        elif weight_type=='cseg':
            wt=self.get_cseg_weight(iobj, icutout)
        elif weight_type=='cseg-canonical':
            wt=self.get_cseg_weight(iobj, icutout, use_canonical_cen=True)
        else:
            raise ValueError("bad weight type '%s'" % weight_type)

        jacobian=ngmix.Jacobian(
            row=jd['row0'],
            col=jd['col0'],
            dudrow=jd['dudrow'],
            dudcol=jd['dudcol'],
            dvdrow=jd['dvdrow'],
            dvdcol=jd['dvdcol'],
        )
        c=self._cat

        scale=jacobian.get_scale()
        x2=c['x2'][iobj]
        y2=c['y2'][iobj]
        T = (x2 + y2)
        flux = c['flux_auto'][iobj]
        meta=dict(
            id=c['id'][iobj],
            T=T,
            flux=flux,
            index=iobj,
            number=c['number'][iobj],
            icut=icutout,
            cutout_index=icutout,
            file_id=c['file_id'][iobj,icutout],
            orig_row=c['orig_row'][iobj, icutout],
            orig_col=c['orig_col'][iobj, icutout],
            orig_start_row=c['orig_start_row'][iobj, icutout],
            orig_start_col=c['orig_start_col'][iobj, icutout],
            orig_end_row=c['orig_end_row'][iobj, icutout],
            orig_end_col=c['orig_end_col'][iobj, icutout],
        )
        obs = ngmix.Observation(
            im,
            weight=wt,
            bmask=bmask,
            meta=meta,
            jacobian=jacobian,
        )
        if False:
            import images
            images.view_mosaic( [obs.image, obs.weight], dims=[800,400] )
            if 'q'==raw_input('hit a key (q to quit): '):
                stop

        return obs

    def get_cseg_weight(self, iobj, icutout, use_canonical_cen=False):
        """
        get the largest circular mask (weight > 0) that does not
        interesect any other objects seg map. If there are no other
        objects in the scene, the regular weight map is returned
        """
        seg = self.get_cutout(iobj, icutout, type='seg')
        weight = self.get_cutout(iobj, icutout, type='weight')
        number = self['number'][iobj]

        wother=np.where( (seg != 0) & (seg != number ) )
        if wother[0].size == 0:
            # no other objects in the stamp
            return weight

        if use_canonical_cen:
            row,col = (np.array(weight.shape)-1.0)/2.0
        else:
            row = self['cutout_row'][iobj, icutout]
            col = self['cutout_col'][iobj, icutout]

        rows, cols = np.mgrid[
            0:weight.shape[0],
            0:weight.shape[1],
        ]
        
        rows = rows.astype('f8')
        cols = cols.astype('f8')

        rows -= row
        cols -= col

        r2 = rows**2 + cols**2

        minr2 = r2[wother].min()

        # now set the weight to zero for radii larger than that
        wkeep = np.where(r2 < minr2)
        new_weight = np.zeros(weight.shape)
        if wkeep[0].size > 0:
            new_weight[wkeep] = weight[wkeep]

        return new_weight

    @property
    def size(self):
        return self._cat.size

    def _check_indices(self, iobj, icutout=None):
        if iobj >= self._cat.size:
            raise ValueError("object index should be within "
                             "[0,%s)" % self._cat.size)

        ncutout=self._cat['ncutout'][iobj]
        if ncutout==0:
            raise ValueError("object %s has no cutouts" % iobj)

        if icutout is not None:
            if icutout >= ncutout:
                raise ValueError("requested cutout index %s for "
                                 "object %s should be in bounds "
                                 "[0,%s)" % (icutout,iobj,ncutout))

class MEDSifier(object):
    def __init__(self,
                 datalist,
                 sx_config=None,
                 meds_config=None):
        """
        very simple MEDS maker for images. Assumes the images are perfectly
        registered and are sky subtracted.

        The images are added together to make a detection image and sep, the
        SExtractor wrapper, is run

        parameters
        ----------
        datalist: list
            List of dicts with entries
                image
                weight
                wcs
            Here wcs is currently assumed to be a simple
            galsim.JacobianWCS; a TODO is to support
            a full WCS
        """
        self.datalist=datalist
        self._set_sx_config(sx_config)
        self._set_meds_config(meds_config)

        self._set_detim()
        self._run_sep()

    def get_multiband_meds(self):
        """
        get a MultiBandMEDS object holding all bands
        """

        mlist=[]
        for band in xrange(len(self.datalist)):
            m=self.get_meds(band)
            mlist.append(m)

        return MultiBandMEDS(mlist)

    def get_meds(self, band):
        """
        get fake MEDS interface to the specified band
        """
        d=self.datalist[band]
        return MEDSInterface(
            d['image'],
            d['weight'],
            self.seg,
            self.bmask,
            self.cat,
        )

    def _get_image_vars(self):
        vars=[]
        for d in self.datalist:
            w=np.where(d['weight'] > 0)
            medw=np.median(d['weight'][w])
            vars.append(1/medw)
        return np.array(vars)

    def _set_detim(self):
        dlist=self.datalist
        nim=len(dlist)

        detim=dlist[0]['image'].copy()
        detim *= 0

        vars = self._get_image_vars()
        weights = 1.0/vars
        wsum = weights.sum()
        detnoise = np.sqrt(1/wsum)

        weights /= wsum

        for i,d in enumerate(dlist):
            detim += d['image']*weights[i]

        self.detim=detim
        self.detnoise=detnoise

    def _run_sep(self):
        import sep
        objs, seg = sep.extract(
            self.detim,
            self.detect_thresh,
            err=self.detnoise,
            segmentation_map=True,
            **self.sx_config
        )

        flux_auto=np.zeros(objs.size)-9999.0
        fluxerr_auto=np.zeros(objs.size)-9999.0
        flux_radius=np.zeros(objs.size)-9999.0
        kron_radius=np.zeros(objs.size)-9999.0

        w,=np.where(
              (objs['a'] >= 0.0)
            & (objs['b'] >= 0.0)
            & between(objs['theta'], -pi/2., pi/2., type='[]')
        )

        if w.size > 0:
            kron_radius[w], krflag = sep.kron_radius(
                self.detim,
                objs['x'][w],
                objs['y'][w],
                objs['a'][w],
                objs['b'][w],
                objs['theta'][w],
                6.0,
            )
            objs['flag'][w] |= krflag

            aper_rad = 2.5*kron_radius
            flux_auto[w], fluxerr_auto[w], flag_auto = \
                sep.sum_ellipse(
                    self.detim,
                    objs['x'][w],
                    objs['y'][w],
                    objs['a'][w],
                    objs['b'][w],
                    objs['theta'][w],
                    aper_rad[w],
                    subpix=1,
                )
            objs['flag'][w] |= flag_auto

            # what we did in DES, but note threshold above
            # is 1 as opposed to wide survey. deep survey
            # was even lower, 0.8?

            # used half light radius
            PHOT_FLUXFRAC = 0.5

            flux_radius[w], frflag = sep.flux_radius(
                self.detim,
                objs['x'][w],
                objs['y'][w],
                6.*objs['a'][w],
                PHOT_FLUXFRAC,
                normflux=flux_auto[w],
                subpix=5,
            )
            objs['flag'][w] |= frflag  # combine flags into 'flag'

        ncut=2 # need this to make sure array
        new_dt=[
            ('id','i8'),
            ('number','i4'),
            ('ncutout','i4'),
            ('kron_radius','f4'),
            ('flux_auto','f4'),
            ('fluxerr_auto','f4'),
            ('flux_radius','f4'),
            ('isoarea_image','f4'),
            ('iso_radius','f4'),
            ('box_size','i4'),
            ('file_id','i8',ncut),
            ('orig_row','f4',ncut),
            ('orig_col','f4',ncut),
            ('orig_start_row','i8',ncut),
            ('orig_start_col','i8',ncut),
            ('orig_end_row','i8',ncut),
            ('orig_end_col','i8',ncut),
            ('cutout_row','f4',ncut),
            ('cutout_col','f4',ncut),
            ('dudrow','f8',ncut),
            ('dudcol','f8',ncut),
            ('dvdrow','f8',ncut),
            ('dvdcol','f8',ncut),
        ]
        cat=eu.numpy_util.add_fields(objs, new_dt)
        cat['id'] = np.arange(cat.size)
        cat['number'] = np.arange(1,cat.size+1)
        cat['ncutout'] = 1
        cat['flux_auto'] = flux_auto
        cat['fluxerr_auto'] = fluxerr_auto
        cat['flux_radius'] = flux_radius

        # TODO make this work with a more general WCS
        wcs=self.datalist[0]['wcs']
        cat['dudrow'][:,0] = wcs.dudy
        cat['dudcol'][:,0] = wcs.dudx
        cat['dvdrow'][:,0] = wcs.dvdy
        cat['dvdcol'][:,0] = wcs.dvdx


        # use the number of pixels in the seg map as the iso area
        for i in xrange(objs.size):
            w=np.where(seg == (i+1))
            cat['isoarea_image'][i] = w[0].size

        cat['iso_radius'] = np.sqrt(cat['isoarea_image'].clip(min=1)/np.pi)


        box_size=self._get_box_sizes(cat)

        half_box_size = box_size//2

        maxrow,maxcol=self.detim.shape

        cat['box_size'] = box_size

        cat['orig_row'][:,0] = cat['y']
        cat['orig_col'][:,0] = cat['x']

        orow = cat['orig_row'][:,0].astype('i4')
        ocol = cat['orig_col'][:,0].astype('i4')

        ostart_row = orow - half_box_size + 1
        ostart_col = ocol - half_box_size + 1
        oend_row   = orow + half_box_size + 1 # plus one for slices
        oend_col   = ocol + half_box_size + 1

        ostart_row.clip(min=0, out=ostart_row)
        ostart_col.clip(min=0, out=ostart_col)
        oend_row.clip(max=maxrow, out=oend_row)
        oend_col.clip(max=maxcol, out=oend_col)

        # could result in smaller than box_size above
        cat['orig_start_row'][:,0] = ostart_row
        cat['orig_start_col'][:,0] = ostart_col
        cat['orig_end_row'][:,0] = oend_row
        cat['orig_end_col'][:,0] = oend_col
        cat['cutout_row'][:,0] = cat['orig_row'][:,0] - cat['orig_start_row'][:,0]
        cat['cutout_col'][:,0] = cat['orig_col'][:,0] - cat['orig_start_col'][:,0]


        self.seg=seg
        self.bmask=np.zeros(seg.shape, dtype='i4')
        self.cat=cat

    def _get_box_sizes(self, cat):

        mconf = self.meds_config

        box_type=mconf['box_type']
        if box_type=='sigma_size':
            sigma_size = self._get_sigma_size(cat)
            row_size = cat['ymax'] - cat['ymin'] + 1
            col_size = cat['xmax'] - cat['xmin'] + 1

            # get max of all three
            box_size = np.vstack((col_size,row_size,sigma_size)).max(axis=0)

        elif box_type=='iso_radius':
            rad_min = mconf['rad_min'] # for box size calculations
            rad_fac = mconf['rad_fac']

            box_padding=mconf['box_padding']
            rad = cat['iso_radius'].clip(min=rad_min)

            box_rad = rad_fac*rad

            box_size = (2*box_rad + box_padding).astype('i4')
        else:
            raise ValueError('bad box type: "%s"' % box_type)

        box_size.clip(
            min=mconf['min_box_size'],
            max=mconf['max_box_size'],
            out=box_size,
        )

        # now put in fft sizes
        bins = [0]
        bins.extend([sze for sze in ALLOWED_BOX_SIZES
                     if sze >= mconf['min_box_size']
                     and sze <= mconf['max_box_size']])

        if bins[-1] != mconf['max_box_size']:
            bins.append(mconf['max_box_size'])

        bin_inds = np.digitize(box_size,bins,right=True)
        bins = np.array(bins)

        box_sizes = bins[bin_inds]
        logger.debug('box sizes: %s' % box_sizes)
        logger.debug('minmax: %s %s' % (box_sizes.min(), box_sizes.max()))

        return box_sizes

    def _get_sigma_size(self, cat):
        """
        "sigma" size, based on flux radius and ellipticity
        """
        mconf = self.meds_config

        ellipticity = 1.0 - cat['b']/cat['a']
        sigma = cat['flux_radius']*2.0/FWHM_FAC
        drad = sigma*mconf['sigma_fac']
        drad = drad*(1.0 + ellipticity)
        drad = np.ceil(drad)
        sigma_size = 2*drad.astype('i4') # sigma size is twice the radius

        return sigma_size

    def _set_sx_config(self, config):
        sx_config={}
        sx_config.update(DEFAULT_SX_CONFIG)

        if config is not None:
            sx_config.update(config)

        if 'filter_kernel' in sx_config:
            sx_config['filter_kernel'] = np.array(sx_config['filter_kernel'])

        self.detect_thresh = sx_config.pop('detect_thresh')
        self.sx_config=sx_config

    def _set_meds_config(self, config):
        meds_config={}
        meds_config.update(DEFAULT_MEDS_CONFIG)

        if config is not None:
            meds_config.update(config)

        self.meds_config=meds_config


def fitpsf(psf_obs):
    am=ngmix.admom.run_admom(psf_obs, 4.0)
    gmix=am.get_gmix()
    return gmix

def get_psf_obs(psfim, jacobian):
    cen=(np.array(psfim.shape)-1.0)/2.0
    j=jacobian.copy()
    j.set_cen(row=cen[0], col=cen[1])

    psf_obs = ngmix.Observation(
        psfim,
        weight=psfim*0+1,
        jacobian=j
    )

    gmix=fitpsf(psf_obs)
    psf_obs.set_gmix(gmix)

    return psf_obs


def test(ntrial=1, dim=2000, show=False):
    import galsim
    import biggles
    import images

    rng=np.random.RandomState()

    nobj_per=4
    nknots=100
    knot_flux_frac=0.001
    nknots_low, nknots_high=1,100

    nband=3
    noises=[0.0005,0.001,0.0015]
    scale=0.263

    psf=galsim.Gaussian(fwhm=0.9)
    dims=64,64
    flux_low, flux_high=0.5,1.5
    r50_low,r50_high=0.1,2.0
    #dims=256,256
    #flux_low, flux_high=50.0,50.0
    #r50_low,r50_high=4.0,4.0

    fracdev_low, fracdev_high=0.001,0.99

    bulge_colors = np.array([0.5, 1.0, 1.5])
    disk_colors = np.array([1.25, 1.0, 0.75])
    knots_colors = np.array([1.5, 1.0, 0.5])

    #bulge_colors /= bulge_colors.sum()
    #disk_colors /= disk_colors.sum()
    #knots_colors /= knots_colors.sum()

    sigma=dims[0]/2.0/4.0*scale
    maxrad=dims[0]/2.0/2.0 * scale

    tm0 = time.time()
    nobj_meas = 0

    for trial in xrange(ntrial):
        print("trial: %d/%d" % (trial+1,ntrial))
        all_band_obj=[]
        for i in xrange(nobj_per):

            nknots=int(rng.uniform(low=nknots_low, high=nknots_high))

            r50=rng.uniform(low=r50_low, high=r50_high)
            flux = rng.uniform(low=flux_low, high=flux_high)

            #dx,dy=rng.uniform(low=-3.0, high=3.0, size=2)
            dx,dy=rng.normal(scale=sigma, size=2).clip(min=-maxrad, max=maxrad)

            g1d,g2d=rng.normal(scale=0.2, size=2).clip(max=0.5)
            g1b=0.5*g1d+rng.normal(scale=0.02)
            g2b=0.5*g2d+rng.normal(scale=0.02)

            fracdev=rng.uniform(low=fracdev_low, high=fracdev_high)

            flux_bulge = fracdev*flux
            flux_disk  = (1-fracdev)*flux
            flux_knots = nknots*knot_flux_frac*flux_disk
            print("fracdev:",fracdev,"nknots:",nknots)

            bulge_obj = galsim.DeVaucouleurs(
                half_light_radius=r50
            ).shear(g1=g1b,g2=g2b)

            disk_obj = galsim.Exponential(
                half_light_radius=r50
            ).shear(g1=g1d,g2=g2d)

            knots_obj = galsim.RandomWalk(
                npoints=nknots,
                profile=disk_obj,
                #half_light_radius=r50
            )#.shear(g1=g1d,g2=g2d)


            band_objs = []
            for band in xrange(nband):
                band_disk=disk_obj.withFlux(flux_disk*disk_colors[band])
                band_bulge=bulge_obj.withFlux(flux_bulge*bulge_colors[band])
                band_knots=knots_obj.withFlux(flux_knots*knots_colors[band])
                #print(band_disk.flux, band_bulge.flux, band_knots.flux)

                #obj = galsim.Sum(band_disk, band_bulge, band_knots).shift(dx=dx, dy=dy)
                obj = galsim.Sum(band_disk, band_bulge).shift(dx=dx, dy=dy)
                #obj = galsim.Sum(band_disk).shift(dx=dx, dy=dy)
                obj=galsim.Convolve(obj, psf)
                band_objs.append( obj )


            all_band_obj.append( band_objs )

        jacob=ngmix.DiagonalJacobian(
            row=0,
            col=0,
            scale=scale,
        )
        wcs=jacob.get_galsim_wcs()
        psfim = psf.drawImage(wcs=wcs).array
        psf_obs=get_psf_obs(psfim, jacob)

        dlist=[]
        for band in xrange(nband):
            band_objects = [ o[band] for o in all_band_obj ]
            obj = galsim.Sum(band_objects)

            im = obj.drawImage(nx=dims[1], ny=dims[0], wcs=wcs).array
            #if band==0:
            #    im = obj.drawImage(scale=scale).array
            #    dims=im.shape
            #else:
            #    im = obj.drawImage(nx=dims[1], ny=dims[0], scale=scale).array
            im = obj.drawImage(nx=dims[1], ny=dims[0], scale=scale).array

            im += rng.normal(scale=noises[band], size=im.shape)
            wt = im*0 + 1.0/noises[band]**2

            dlist.append(
                dict(
                    image=im,
                    weight=wt,
                    wcs=wcs,
                )
            )


        mer=MEDSifier(dlist)

        mg=mer.get_meds(0)
        mr=mer.get_meds(1)
        mi=mer.get_meds(2)
        nobj=mg.size
        print("        found",nobj,"objects")
        nobj_meas += nobj

        #3imlist=[]
        list_of_obs=[]
        for i in xrange(nobj):

            img=mg.get_cutout(i,0)
            imr=mr.get_cutout(i,0)
            imi=mi.get_cutout(i,0)

            gobslist=mg.get_obslist(i,weight_type='uberseg')
            robslist=mr.get_obslist(i,weight_type='uberseg')
            iobslist=mi.get_obslist(i,weight_type='uberseg')
            mbo=ngmix.MultiBandObsList()
            mbo.append(gobslist)
            mbo.append(robslist)
            mbo.append(iobslist)

            list_of_obs.append(mbo)

        for mbo in list_of_obs:
            for obslist in mbo:
                for obs in obslist:
                    obs.set_psf(psf_obs)

        prior=moflib.get_mof_prior(list_of_obs, "bdf", rng)
        mof_fitter=moflib.MOFStamps(
            list_of_obs,
            "bdf",
            prior=prior,
        )
        band=2
        guess=moflib.get_stamp_guesses(list_of_obs, band, "bdf", rng)
        mof_fitter.go(guess)

        if show:
            # corrected images
            tab=biggles.Table(1,2)
            rgb=images.get_color_image(
                #imi*fac,imr*fac,img*fac,
                dlist[2]['image'].transpose(),
                dlist[1]['image'].transpose(),
                dlist[0]['image'].transpose(),
                nonlinear=0.1,
            )
            rgb *= 1.0/rgb.max()
         
            tab[0,0] = images.view_mosaic(
                [rgb,
                 mer.seg,
                 mer.detim],
                titles=['image','seg','detim'],
                show=False,
                #dims=[dim, dim],
            )


            imlist=[]
            for iobj, mobs in enumerate(list_of_obs):
                cmobs = mof_fitter.make_corrected_obs(iobj)

                gim=images.make_combined_mosaic(
                    [mobs[0][0].image, cmobs[0][0].image],
                )
                rim=images.make_combined_mosaic(
                    [mobs[1][0].image, cmobs[1][0].image],
                )
                iim=images.make_combined_mosaic(
                    [mobs[2][0].image, cmobs[2][0].image],
                )

                rgb=images.get_color_image(
                    iim.transpose(),
                    rim.transpose(),
                    gim.transpose(),
                    nonlinear=0.1,
                )
                rgb *= 1.0/rgb.max()
                imlist.append(rgb)


            plt=images.view_mosaic(imlist,show=False)
            tab[0,1]=plt
            tab.show(width=dim*2, height=dim)

            if ntrial > 1:
                if 'q'==raw_input("hit a key: "):
                    return

    total_time=time.time()-tm0
    print("time per group:",total_time/ntrial)
    print("time per object:",total_time/nobj_meas)
