"""
copied from meas_extensions_ngmix and modified
"""
import numpy as np
import ngmix
import lsst.log
import lsst.afw.image as afw_image
from lsst.pex.exceptions import InvalidParameterError, LogicError
import lsst.geom as geom

from . import util


class MBObsMissingDataError(Exception):
    """
    Some number was out of range
    """

    def __init__(self, value):
        super(MBObsMissingDataError, self).__init__(value)
        self.value = value

    def __str__(self):
        return repr(self.value)


class MBObsExtractor(object):
    """
    class to extract observations from the images

    parameters
    ----------
    config: dict or dict like
        A dictionary configuration
    exposures: list of exposures
        List of ExposureF
    sources: records
        records from detection
    psfs: optional
        List of psf images
    """

    def __init__(self, *, config, exposures, sources, psfs=None):
        self.config = config
        self.exposures = exposures
        self.sources = sources
        self.psfs = psfs

        self.log = lsst.log.Log.getLogger("MBObsExtractor")

        self._verify()

    def get_mbobs_list(self, weight_type='weight'):
        """
        Get a list of `MultiBandObsList` for all or a set of objects.

        Parameters
        ----------
        weight_type: string, optional
            Currently only 'weight' is supported

        Returns
        -------
        mbobs_list : list of ngmix.MultiBandObsList
            The list of `MultiBandObsList`s for the requested objects.
        """

        list_of_obs = []

        for rec in self.sources:
            mbobs = self._get_mbobs(rec, weight_type=weight_type)
            list_of_obs.append(mbobs)

        return list_of_obs

    def _get_mbobs(self, rec, weight_type='weight'):
        """
        make an ngmix.MultiBandObsList for input to the fitter

        parameters
        ----------
        images: dict
            A dictionary of image objects
        rec: object record
            TODO I don't actually know what class this is

        returns
        -------
        mbobs: ngmix.MultiBandObsList
            ngmix multi-band observation list
        weight_type: string, optional
            Currently only 'weight' is supported
        """

        assert weight_type == 'weight'

        # rec = self.sources[iobj]

        mbobs = ngmix.MultiBandObsList()

        for im_index, imf in enumerate(self.exposures):
            # TODO: run noise replacers here

            bbox = self._get_bbox(rec, imf)
            subim = _get_padded_sub_image(imf, bbox)

            if self.psfs is not None:
                psf_im = self.psfs[im_index]
            else:
                psf_im = None

            obslist = ngmix.ObsList()
            try:
                obs = self._extract_obs(subim, rec, psf_im=psf_im)
                obslist.append(obs)
            except ngmix.GMixFatalError as err:
                self.log.info(str(err))

            mbobs.append(obslist)

        if len(mbobs[0]) > 0:
            self.log.debug('    stamp shape: %s' % str(mbobs[0][0].image.shape))

        return mbobs

    def _get_bbox(self, rec, imobj):
        """
        get the bounding box for this object

        TODO fine tune the bounding box algorithm

        parameters
        ----------
        rec: object record
            TODO I don't actually know what class this is

        returns
        -------
        bbox:
            TODO I don't actually know what class this is
        """

        try:
            stamp_radius, stamp_size = self._compute_stamp_size(rec)
            bbox = _project_box(rec, imobj.getWcs(), stamp_radius)
        except LogicError as err:
            self.log.debug(str(err))
            bbox = rec.getFootprint().getBBox()

        return bbox

    def _compute_stamp_size(self, rec):
        """
        calculate a stamp radius for the input object, to
        be used for constructing postage stamp sizes
        """
        sconf = self.config['stamps']

        min_radius = sconf['min_stamp_size']/2
        max_radius = sconf['max_stamp_size']/2

        quad = rec.getShape()
        T = quad.getIxx() + quad.getIyy()
        if np.isnan(T):
            T = 4.0

        sigma = np.sqrt(T/2.0)
        radius = sconf['sigma_factor']*sigma

        if radius < min_radius:
            radius = min_radius
        elif radius > max_radius:
            radius = max_radius

        radius = int(np.ceil(radius))
        stamp_size = 2*radius+1

        return radius, stamp_size

    def _extract_obs(self, imobj_sub, rec, psf_im=None):
        """
        convert an image object into an ngmix.Observation, including
        a psf observation

        parameters
        ----------
        imobj: an image object
            TODO I don't actually know what class this is
        rec: an object record
            TODO I don't actually know what class this is

        returns
        --------
        obs: ngmix.Observation
            The Observation, including
        """

        im = imobj_sub.image.array
        wt = self._extract_weight(imobj_sub)
        maskobj = imobj_sub.mask
        bmask = maskobj.array
        jacob = self._extract_jacobian(imobj_sub, rec)

        # cen = rec.getCentroid()
        orig_cen = imobj_sub.getWcs().skyToPixel(rec.getCoord())

        if psf_im is None:
            psf_im = self._extract_psf_image(imobj_sub, orig_cen)

        # fake the psf pixel noise
        psf_err = psf_im.max()*0.0001
        psf_wt = psf_im*0 + 1.0/psf_err**2

        # use canonical center for the psf
        psf_cen = (np.array(psf_im.shape)-1.0)/2.0
        psf_jacob = jacob.copy()
        psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

        # we will have need of the bit names which we can only
        # get from the mask object
        # this is sort of monkey patching, but I'm not sure of
        # a better solution
        meta = {'maskobj': maskobj}

        psf_obs = ngmix.Observation(
            psf_im,
            weight=psf_wt,
            jacobian=psf_jacob,
        )
        obs = ngmix.Observation(
            im,
            weight=wt,
            bmask=bmask,
            jacobian=jacob,
            psf=psf_obs,
            meta=meta,
        )

        return obs

    def _extract_psf_image(self, stamp, orig_pos):
        """
        get the psf associated with this stamp

        coadded psfs are generally not square, so we will
        trim it to be square and preserve the center to
        be at the new canonical center
        """
        try:
            psfobj = stamp.getPsf()
            psfim = psfobj.computeKernelImage(orig_pos).array
        except InvalidParameterError:
            raise MBObsMissingDataError("could not reconstruct PSF")

        psfim = np.array(psfim, dtype='f4', copy=False)

        psfim = util.trim_odd_image(psfim)
        return psfim

    def _extract_jacobian(self, imobj, rec):
        """
        extract an ngmix.Jacobian from the image object
        and object record

        imobj: an image object
            TODO I don't actually know what class this is
        rec: an object record
            TODO I don't actually know what class this is

        returns
        --------
        Jacobian: ngmix.Jacobian
            The local jacobian
        """

        xy0 = imobj.getXY0()

        orig_cen = imobj.getWcs().skyToPixel(rec.getCoord())
        if np.isnan(orig_cen.getY()):
            self.log.debug('falling back on integer location')
            # fall back to integer pixel location
            peak = rec.getFootprint().getPeaks()[0]
            orig_cenI = peak.getI()
            orig_cen = geom.Point2D(
                x=orig_cenI.getX(),
                y=orig_cenI.getY(),
            )
            # x, y = peak.getIx(), peak.getIy()

        cen = orig_cen - geom.Extent2D(xy0)
        row = cen.getY()
        col = cen.getX()

        wcs = imobj.getWcs().linearizePixelToSky(
            orig_cen,
            geom.arcseconds,
        )
        jmatrix = wcs.getLinear().getMatrix()

        jacob = ngmix.Jacobian(
            row=row,
            col=col,
            dudcol=jmatrix[0, 0],
            dudrow=jmatrix[0, 1],
            dvdcol=jmatrix[1, 0],
            dvdrow=jmatrix[1, 1],
        )

        self.log.debug("jacob: %s" % repr(jacob))
        return jacob

    def _extract_weight(self, imobj):
        """
        TODO get the estimated sky variance rather than this hack
        TODO should we zero out other bits?

        extract a weight map

        Areas with NO_DATA will get zero weight.

        Because the variance map includes the full poisson variance, which
        depends on the signal, we instead extract the median of the parts of
        the image without NO_DATA set

        parameters
        ----------
        imobj: an image object
            TODO I don't actually know what class this is
        """
        var_image = imobj.variance.array
        maskobj = imobj.mask
        mask = maskobj.array

        weight = var_image.copy()

        weight[:, :] = 0

        bitnames_to_ignore = self.config['stamps']['bits_to_ignore_for_weight']

        bits_to_ignore = util.get_ored_bits(maskobj, bitnames_to_ignore)

        wuse = np.where(
            (var_image > 0) &
            ((mask & bits_to_ignore) == 0)
        )

        if wuse[0].size > 0:
            medvar = np.median(var_image[wuse])
            weight[:, :] = 1.0/medvar
        else:
            self.log.debug('    weight is all zero, found none that passed cuts')
            # _print_bits(maskobj, bitnames_to_ignore)

        bitnames_to_null = self.config['stamps']['bits_to_null']
        if len(bitnames_to_null) > 0:
            bits_to_null = util.get_ored_bits(maskobj, bitnames_to_null)
            wnull = np.where((mask & bits_to_null) != 0)
            if wnull[0].size > 0:
                self.log.debug('    nulling %d in weight' % wnull[0].size)
                weight[wnull] = 0.0

        return weight

    def _verify(self):
        """
        check for consistency between the images.

        .. todo::
           An assertion is currently used, we may want to raise an
           appropriate exception.
        """

        assert isinstance(self.exposures, list)

        if self.psfs is not None:
            assert isinstance(self.psfs, list)
            assert len(self.psfs) == len(self.exposures)

        for i, imf in enumerate(self.exposures):
            if i == 0:
                xy0 = imf.getXY0()
            else:
                assert xy0 == imf.getXY0(),\
                    "all images must have same reference position"


def _project_box(source, wcs, radius):
    """
    create a box for the input source
    """
    pixel = geom.Point2I(wcs.skyToPixel(source.getCoord()))
    box = geom.Box2I()
    box.include(pixel)
    box.grow(radius)
    return box


def _get_padded_sub_image(original, bbox):
    """
    extract a sub-image, padded out when it is not contained
    """
    region = original.getBBox()

    if region.contains(bbox):
        return original.Factory(original, bbox, afw_image.PARENT, True)

    result = original.Factory(bbox)
    bbox2 = geom.Box2I(bbox)
    bbox2.clip(region)
    if isinstance(original, afw_image.Exposure):
        result.setPsf(original.getPsf())
        result.setWcs(original.getWcs())
        result.setPhotoCalib(original.getPhotoCalib())
        # result.image.array[:, :] = float("nan")
        result.image.array[:, :] = 0.0
        result.variance.array[:, :] = float("inf")
        result.mask.array[:, :] = np.uint16(result.mask.getPlaneBitMask("NO_DATA"))
        sub_in = afw_image.MaskedImageF(original.maskedImage, bbox=bbox2,
                                        origin=afw_image.PARENT, deep=False)
        result.maskedImage.assign(sub_in, bbox=bbox2, origin=afw_image.PARENT)
    elif isinstance(original, afw_image.ImageI):
        result.array[:, :] = 0
        sub_in = afw_image.ImageI(original, bbox=bbox2,
                                  origin=afw_image.PARENT, deep=False)
        result.assign(sub_in, bbox=bbox2, origin=afw_image.PARENT)
    else:
        raise ValueError("Image type not supported")
    return result


def _print_bits(maskobj, bitnames):
    mask = maskobj.array
    for ibit, bitname in enumerate(bitnames):
        bitval = maskobj.getPlaneBitMask(bitname)
        w = np.where((mask & bitval) != 0)
        if w[0].size > 0:
            print('%s %d %d/%d' % (bitname, bitval, w[0].size, mask.size))
