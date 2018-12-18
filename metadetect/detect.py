"""

Code to take a set of images, for example coadds in different bands, run
detection on them and produce a MEDS like interface to get postage stamps

TODO factor out the detection code so we could, for example, drop in the LSST
DM stack detection

"""
from __future__ import print_function
import logging
import numpy as np
from numpy import pi
import esutil as eu
from esutil.numpy_util import between

from ngmix.medsreaders import NGMixMEDS, MultiBandNGMixMEDS

from . import defaults

logger = logging.getLogger(__name__)

FWHM_FAC = 2*np.sqrt(2*np.log(2))


class MEDSInterface(NGMixMEDS):
    """
    Wrap a full image with a MEDS-like interface.

    Parameters
    ----------
    obs: ngmix.observation.Observation
        The observation with the image to wrap.
    seg: np.array
        The segmentation map for the observation.
    cat: structured np.array
        The catalog for the MEDS data format.
    psf_rec_func: function, optional
        A function with call signature `psf_rec_func(row, col)`. If not `None`,
        then this function will be called to get the PSF image at `(row, col)`
        for each cutout.
    """
    def __init__(self, obs, seg, cat, psf_rec_func=None):
        self.obs = obs
        self.seg = seg
        self._image_types = (
            'image', 'weight', 'seg', 'bmask', 'noise')
        self._cat = cat
        self.psf_rec_func = psf_rec_func

    def get_psf(self, iobj, icut):
        """
        get an image of the psf
        """
        if self.psf_rec_func is None:
            return self.obs.psf.image.copy()
        else:
            assert icut == 0
            row = self._cat['orig_row'][iobj, icut]
            col = self._cat['orig_col'][iobj, icut]
            return self.psf_rec_func(row, col)

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

        self._check_indices(iobj, icutout=icutout)

        if type == 'psf':
            return self.get_psf(iobj, icutout)

        im = self._get_type_image(type)
        dims = im.shape

        c = self._cat
        orow = c['orig_start_row'][iobj, icutout]
        ocol = c['orig_start_col'][iobj, icutout]
        bsize = c['box_size'][iobj]

        orow_box, row_box = self._get_clipped_boxes(dims[0], orow, bsize)
        ocol_box, col_box = self._get_clipped_boxes(dims[1], ocol, bsize)

        read_im = im[orow_box[0]:orow_box[1],
                     ocol_box[0]:ocol_box[1]]

        subim = np.zeros((bsize, bsize), dtype=im.dtype)
        subim += defaults.DEFAULT_IMAGE_VALUES[type]

        subim[row_box[0]:row_box[1],
              col_box[0]:col_box[1]] = read_im

        return subim

    def _get_type_image(self, type):
        if type not in self._image_types:
            raise ValueError("bad cutout type: '%s'" % type)

        if type == 'image':
            data = self.obs.image
        elif type == 'weight':
            data = self.obs.weight
        elif type == 'bmask':
            data = self.obs.bmask
        elif type == 'noise':
            data = self.obs.noise
        elif type == 'seg':
            data = self.seg

        return data

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
        diff = im_max - obox[1]
        if diff < 0:
            obox[1] = im_max
            box[1] = box[1] + diff

        return obox, box


class MEDSifier(object):
    def __init__(self,
                 mbobs,
                 sx_config,
                 meds_config,
                 wcs_jacobian_func=None,
                 pos_transform_func=None):
        """
        very simple MEDS maker for images. Assumes the images are perfectly
        registered and are sky subtracted, with constant PSF and WCS.

        The images are added together to make a detection image and sep, the
        SExtractor wrapper, is run

        parameters
        ----------
        mbobs: ngmix.MultiBandObsList
            The data
        sx_config: dict, optional
            Dict holding sep extract parameters
        meds_config: dict, optional
            Dict holding MEDS parameters
        wcs_jacobian_func: function, optional
            A function with call signature `wcs_jacobian_func(row, col)` that
            returns a dictionary with keys {'dudrow', 'dudcol',
            'dvdrow', 'dvdcol'} and the corresponding values. If `None`, then
            the jacobian of the input mbobs is used.
        pos_transform_func: function, optional
            A function to transform the detected positions. The call signature
            should be `pos_transform_func(x, y)` and it should return the new
            position as `(x_new, y_new)`. If None, then no transformation is
            done.
        """
        self.mbobs=mbobs
        self.nband=len(mbobs)
        assert len(mbobs[0])==1,'multi-epoch is not supported'
        self.wcs_jacobian_func = wcs_jacobian_func
        self.pos_transform_func = pos_transform_func

        self._set_sx_config(sx_config)
        self._set_meds_config(meds_config)

        self._set_detim()
        self._run_sep()

    def get_multiband_meds(self):
        """
        get a MultiBandMEDS object holding all bands
        """

        mlist=[]
        for band in range(self.nband):
            m=self.get_meds(band)
            mlist.append(m)

        return MultiBandNGMixMEDS(mlist)

    def get_meds(self, band):
        """
        get fake MEDS interface to the specified band
        """
        obslist=self.mbobs[band]
        obs = obslist[0]
        return MEDSInterface(
            obs,
            self.seg,
            self.cat,
        )

    def _get_image_vars(self):
        vars=[]
        for obslist in self.mbobs:
            obs=obslist[0]
            weight=obs.weight
            w=np.where(weight > 0)
            medw=np.median(weight[w])
            vars.append(1/medw)
        return np.array(vars)

    def _set_detim(self):

        detim=self.mbobs[0][0].image.copy()
        detim *= 0

        vars = self._get_image_vars()
        weights = 1.0/vars
        wsum = weights.sum()
        detnoise = np.sqrt(1/wsum)

        weights /= wsum

        for i,obslist in enumerate(self.mbobs):
            obs=obslist[0]
            detim += obs.image*weights[i]

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

        # use the number of pixels in the seg map as the iso area
        for i in range(objs.size):
            w=np.where(seg == (i+1))
            cat['isoarea_image'][i] = w[0].size

        cat['iso_radius'] = np.sqrt(cat['isoarea_image'].clip(min=1)/np.pi)

        if cat.size > 0:

            box_size=self._get_box_sizes(cat)

            half_box_size = box_size//2

            maxrow,maxcol=self.detim.shape

            cat['box_size'] = box_size

            if self.pos_transform_func is not None:
                pos_new = self.pos_transform_func(cat['x'], cat['y'])
                cat['x'] = pos_new[0]
                cat['y'] = pos_new[1]

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

        if self.wcs_jacobian_func is None:
            jacob = self.mbobs[0][0].jacobian
            cat['dudrow'][:, 0] = jacob.dudrow
            cat['dudcol'][:, 0] = jacob.dudcol
            cat['dvdrow'][:, 0] = jacob.dvdrow
            cat['dvdcol'][:, 0] = jacob.dvdcol
        else:
            for i in range(objs.size):
                jacob = self.wcs_jacobian_func(
                    cat['orig_row'][i, 0], cat['orig_col'][i, 0])
                cat['dudcol'][i, 0] = jacob['dudcol']
                cat['dudrow'][i, 0] = jacob['dudrow']
                cat['dvdcol'][i, 0] = jacob['dvdcol']
                cat['dvdrow'][i, 0] = jacob['dvdrow']

        self.seg=seg
        self.cat=cat

    def _get_box_sizes(self, cat):
        if cat.size == 0:
            return []

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
        bins.extend([sze for sze in defaults.ALLOWED_BOX_SIZES
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

    def _set_sx_config(self, sx_config_in):

        sx_config={}
        sx_config.update(sx_config_in)
        sx_config['filter_kernel'] = np.array(sx_config['filter_kernel'])

        # this isn't a keyword
        self.detect_thresh = sx_config.pop('detect_thresh')
        self.sx_config=sx_config

    def _set_meds_config(self, meds_config):
        self.meds_config=meds_config
