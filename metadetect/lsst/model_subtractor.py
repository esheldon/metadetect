import logging
from contextlib import contextmanager
from lsst.afw.image import MultibandExposure
import lsst.afw.image as afw_image
import lsst.geom as geom
from lsst.afw.table import SourceCatalog
from lsst.pex.exceptions import LengthError
# from lsst.meas.extensions.scarlet.io.utils import updateCatalogFootprints
from . import util

import numpy as np
import lsst.scarlet.lite as scl
from lsst.afw.detection import HeavyFootprintF, makeHeavyFootprint
from lsst.afw.detection.multiband import MultibandFootprint
from lsst.afw.image import Mask, MaskedImage, MultibandImage
from lsst.afw.geom import SpanSet
from lsst.afw.detection import Footprint as afwFootprint
from lsst.afw.image import Image as afwImage

LOG = logging.getLogger('lsst_model_subtractor')


class ModelSubtractor(object):
    """
    Takes in an image, catalog and models and produces an image with all models
    subtracted.  Models can be then inserted one at a time and a postage stamp
    extracted.  Stamps for the model or origina image can also be extracted.

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        A representation of the multi-band data set.
        Create one of these with
            descwl_model_subtractor.get_mbexp(exposure_list)
    sources: lsst.afw.table.SourceCatalog
        This is the output of the detection

    model_data: LsstScarletBlendData
        The structure holding model information

    Examples
    ---------

    subtractor = ModelSubtractor(exposure, sources, model_data)

    # add back one model; since the image had data-model this restores the
    # pixels for the object of interest, but with models of other objects
    # subtracted

    for i, source in subtractor.children():
        source_id = source.getId()
        with subtractor.add_source(source_id):
            stamp = subtractor.get_stamp(source_id, stamp_size=48)

    # model of the entire data set as a MultibandExposure
    full_model = subtractor.get_full_model()

    # model of one source
    model = subtractor.get_model(source_id, stamp_size=48)
    """

    def __init__(self, mbexp, sources, model_data):
        assert isinstance(mbexp, MultibandExposure), (
            f'For input mbexp, expected MultibandExposure, got {type(mbexp)}'
        )
        assert isinstance(sources, SourceCatalog), (
            f'For input sources, expected SourceCatalog, got {type(sources)}'
        )

        self.orig = mbexp
        self.model_data = model_data

        # we will work with this copy rather than the original
        self.mbexp = util.copy_mbexp(mbexp)

        self.sources = sources

        self.bands = mbexp.filters

        self._build_subtracted_image()

    def children(self):
        """
        Generator that yields children Sources
        """
        for sid in self.child_ids():
            yield self.sources.find(sid)

    def child_ids(self):
        """
        Generator that yields child ids
        """
        for sid in self.heavies:
            yield sid

    def check_source_id(self, source_id):
        """
        Check the source id is a child and has a model
        """
        if source_id not in self.heavies:
            raise ValueError(
                f'source {source_id} is not in the child source list',
            )

    @contextmanager
    def add_source(self, source_id):
        """
        Open a with context with one model temporarily added back to the image.

        This restores the pixels for the object of interest, with models of
        other objects still subtracted.

        On exit from the context, the model is again subtracted

        with subtractor.add_source(source_id):
            # do something with subtractor.mbexp

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()

        Yields
        -------
        The MultiBandExposure object, although usually the point is
        to extract a stamp
        """
        self.check_source_id(source_id)

        self._add_or_subtract_source(source_id, 'add')
        # self._add_or_subtract_source_new(source_id, 'add')
        try:
            yield self.mbexp
        finally:
            self._add_or_subtract_source(source_id, 'subtract')
            # self._add_or_subtract_source_new(source_id, 'subtract')

    def _add_or_subtract_source(self, source_id, type):
        mbexp = self.mbexp
        bbox = self.bboxes[source_id]
        heavy = self.heavies[source_id]

        for band in self.bands:

            if type == 'add':
                heavy[band].addTo(mbexp[band].image[bbox])
            else:
                heavy[band].subtractFrom(mbexp[band].image[bbox])

    def get_stamp(
        self,
        source_id,
        stamp_size=None,
        clip=False,
        type='deblended',
    ):
        """
        Get a postage stamp exposure at the location of the specified source.
        The pixel data are copied.

        If you want the object to be in the image, use this method within
        an add_source context

        with subtractor.add_source(source_id):
            stamp = subtractor.get_stamp(source_id)

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()
        stamp_size: int, optional
            If sent, a bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

            Only relevant if stamp_size is sent.  Default False
        type: str, optional
            'deblended', 'original', 'model'.  Default is 'deblended'.

            'deblended' means whatever is in the current subtracted images.
                If the user is in the add_source() context it will contain
                that source data because the model will have been added back in

            'original' means a stamp from the original data, without models
                subtracted

            'model' means the model for object
                You can also use get_model() to get the model

        Returns
        -------
        ExposureF
        """

        assert type in ['deblended', 'original', 'model'], (
            'type must be one of deblended, original or model'
        )

        if type == 'model':
            return self.get_model(source_id, stamp_size=stamp_size, clip=clip)

        self.check_source_id(source_id)

        bbox = self.get_bbox(source_id, stamp_size=stamp_size, clip=clip)

        if type == 'original':
            mbexp = self.orig
        else:
            mbexp = self.mbexp

        exposures = [mbexp[band][bbox] for band in self.bands]
        # return MultibandExposure.fromExposures(self.bands, exposures)
        return util.get_mbexp(exposures)

    def get_mbobs(
        self,
        source_id,
        stamp_size=None,
        clip=False,
        type='deblended',
    ):
        """
        Get a ngmix.MultiBandObsList centered at the location of the
        specified source.  The pixel data are copied.

        If you want the object to be in the image, use this method within
        an add_source context

        with subtractor.add_source(source_id):
            stamp = subtractor.get_mbobs(source_id)

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()
        stamp_size: int
            If sent, a bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

            Only relevant if stamp_size is sent.  Default False
        type: str, optional
            'deblended', 'original', 'model'.  Default is 'deblended'.

            'deblended' means whatever is in the current subtracted images.
                If the user is in the add_source() context it will contain
                that source data because the model will have been added back in

            'original' means a stamp from the original data, without models
                subtracted

            'model' means the model for object
                You can also use get_model() to get the model

        Returns
        -------
        ngmix.MultiBandObsList
        """

        assert type in ['deblended', 'original', 'model'], (
            'type must be one of deblended, original or model'
        )
        import ngmix

        if type == 'model':
            return self.get_model(source_id, stamp_size=stamp_size, clip=clip)

        self.check_source_id(source_id)

        bbox = self.get_bbox(source_id, stamp_size=stamp_size, clip=clip)

        if type == 'original':
            mbexp = self.orig
        else:
            mbexp = self.mbexp

        source = self.sources.find(source_id)
        mbobs = ngmix.MultiBandObsList()

        for band in mbexp.bands:
            subexp = mbexp[band][bbox]
            obs = util.extract_obs(exp=subexp, source=source)

            obslist = ngmix.ObsList(meta={'band': band})
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs

    def get_model(self, source_id, stamp_size=None, clip=False):
        """
        Get a postage stamp exposure at the location of the specified source,
        containing the model rather than data.

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()
        stamp_size: int
            If sent, a bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

            Only relevant if stamp_size is sent.  Default False

        Returns
        -------
        ExposureF
        """

        self.check_source_id(source_id)

        heavies = self.heavies

        heavy = heavies[source_id]

        exposures = []
        for band in self.bands:
            im = heavy[band].extractImage()
            masked_im = afw_image.MaskedImageF(im.clone())
            exp = afw_image.ExposureF(masked_im)
            exp.setFilter(afw_image.FilterLabel(band))
            exposures.append(exp)

        return util.get_mbexp(exposures)

    def get_full_model(self):
        """
        Get a full model image of all sources.

        Returns
        -------
        ExposureF
        """

        model = util.copy_mbexp(self.mbexp, clear=True)

        for sid, heavy in self.heavies.items():
            bbox = self.bboxes[sid]
            for band in self.bands:
                heavy[band].addTo(model[band].image[bbox])

        return model

    def get_bbox(self, source_id, stamp_size=None, clip=False):
        """
        Get a bounding box at the location of the specified source.

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()
        stamp_size: int
            If sent, a bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

            Only relevant if stamp_size is sent.  Default False

        Returns
        -------
        lsst.geom.Box2I
        """

        self.check_source_id(source_id)

        if stamp_size is not None:
            fp = self.footprints[source_id]
            peak = fp.getPeaks()[0]

            # note we previously had -0.5 on each of these based on Bob's code
            x_peak, y_peak = peak.getIx(), peak.getIy()

            bbox = geom.Box2I(
                geom.Point2I(x_peak, y_peak),
                geom.Extent2I(1, 1),
            )
            bbox.grow(stamp_size // 2)

            exp_bbox = self.mbexp.getBBox()
            if clip:
                bbox.clip(exp_bbox)
            else:
                if not exp_bbox.contains(bbox):
                    raise LengthError(
                        f'requested stamp size {stamp_size} for source '
                        f'{source_id} does not fit into the exposoure.  '
                        f'Use clip=True to clip the bbox to fit'
                    )

        else:
            parent_id, fp = self.footprints[source_id]
            bbox = fp.getBBox()

        return bbox

    def _build_subtracted_image(self):
        self.heavies = {}
        self.footprints = {}
        self.bboxes = {}

        model_data = self.model_data

        bands = model_data.metadata["bands"]
        model_psf = model_data.metadata["model_psf"]
        observed_psf = model_data.metadata["psf"]

        for full_blend_data in model_data.blends.values():
            for blend_data in full_blend_data.children.values():
                blend = blend_data.minimal_data_to_blend(
                    model_psf=model_psf[None],
                    psf=observed_psf,
                    bands=bands,
                )
                for scl_source in blend.sources:
                    # this id is same as catalog source getId()
                    sid = scl_source.metadata['id']
                    source = self.sources.find(sid)
                    footprint = source.getFootprint()
                    bbox = footprint.getBBox()

                    heavy = scarletModelToHeavy(source=scl_source, blend=blend)

                    for band in bands:
                        heavy[band].subtractFrom(self.mbexp[band].image[bbox])

                    self.bboxes[sid] = bbox
                    self.heavies[sid] = heavy
                    self.footprints[sid] = footprint


def scarletModelToHeavy(
    source: scl.Source,
    blend: scl.Blend,
    useFlux=False,
) -> HeavyFootprintF | MultibandFootprint:
    """
    Convert a scarlet_lite model to a `HeavyFootprintF` or
    `MultibandFootprint`.

    This is a copy of the code from scarlet meas extension with
    a bug fix for nbands in multi epoch

    Parameters
    ----------
    source:
        The source to convert to a `HeavyFootprint`.
    blend:
        The `Blend` object that contains information about
        the observation, PSF, etc, used to convolve the
        scarlet model to the observed seeing in each band.
    useFlux:
        Whether or not to re-distribute the flux from the image
        to conserve flux.

    Returns
    -------
    heavy:
        The footprint (possibly multiband) containing the model for the source.
    """
    # We want to convolve the model with the observed PSF,
    # which means we need to grow the model box by the PSF to
    # account for all of the flux after convolution.

    # Get the PSF size and radii to grow the box
    py, px = blend.observation.psfs.shape[1:]
    dh = py // 2
    dw = px // 2

    if useFlux:
        bbox = source.flux_weighted_image.bbox
    else:
        bbox = source.bbox.grow((dh, dw))
    # Only use the portion of the convolved model that fits in the image
    overlap = bbox & blend.observation.bbox
    # Load the full multiband model in the larger box
    if useFlux:
        # The flux weighted model is already convolved, so we just load it
        model = source.get_model(use_flux=True).project(bbox=overlap)
    else:
        model = source.get_model().project(bbox=overlap)
        # Convolve the model with the PSF in each band
        # Always use a real space convolution to limit artifacts
        model = blend.observation.convolve(model, mode="real")

    # Update xy0 with the origin of the sources box
    xy0 = geom.Point2I(model.yx0[-1], model.yx0[-2])
    # Create the spans for the footprint
    valid = np.max(model.data, axis=0) != 0
    valid = Mask(valid.astype(np.int32), xy0=xy0)
    spans = SpanSet.fromMask(valid)

    # Create the MultibandHeavyFootprint and
    # add the location of the source to the peak catalog.
    foot = afwFootprint(spans)
    foot.addPeak(source.center[1], source.center[0], np.max(model.data))
    if model.n_bands == 1:
        image = afwImage(
            array=model.data[0],
            xy0=valid.getBBox().getMin(),
            dtype=model.dtype,
        )
        maskedImage = MaskedImage(image, dtype=model.dtype)
        heavy = makeHeavyFootprint(foot, maskedImage)
    else:
        # BUG fix:  was blend.bands
        model = MultibandImage(
            blend.observation.bands, model.data, valid.getBBox(),
        )
        heavy = MultibandFootprint.fromImages(
            blend.observation.bands, model, footprint=foot
        )
    return heavy
