import logging
from contextlib import contextmanager
from lsst.afw.image import MultibandExposure
import lsst.afw.image as afw_image
import lsst.geom as geom
import lsst.afw.detection as afw_det
from lsst.afw.table import SourceCatalog
from lsst.pex.exceptions import LengthError
from lsst.meas.extensions.scarlet.io.utils import updateCatalogFootprints
from . import util

LOG = logging.getLogger('lsst_model_subtractor')


class ModelSubtractor(object):
    """
    Create an image with all models subtracted, which is stored in the .mbexp
    attribute

    Provides a method to add back a source.  This works in a context manager to
    maintain data consistency.

    Provides methods to get a postage stamp for a deblended source.  One can
    also get a stamp for the model or the original image.

    You an also generate the full model of the image.

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        A representation of the multi-band data set.
        Create one of these with
            descwl_model_subtractor.get_mbexp(exposure_list)
    sources: lsst.afw.table.SourceCatalog
        This is the output of the detection

    TODO
    ----
        - This code would use half the memory if heavy footprints supported
        operations like .addTo(image) or .subtractFrom(image) which should be
        essentially the same as the code in .insert

    Examples
    ---------

    subtractor = ModelSubtractor(exposure, sources)

    # add back one model; since the image had data-model this restores the
    # pixels for the object of interest, but with models of other objects
    # subtracted

    with subtractor.add_source(source_id):
        # full MultibandExposure is in subtractor.mbexp

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

        # we will work with this copy rather than the original
        self.mbexp = util.copy_mbexp(mbexp)

        # we need a scratch array because heavy footprings don't
        # have addTo or subtractFrom methods
        self.scratch = util.copy_mbexp(mbexp, clear=True)

        self.sources_orig = sources

        self.bands = mbexp.filters

        # make a deep copy of sources for each band, which will hold models
        self.band_sources = {
            band: sources.copy(deep=True) for band in self.bands
        }

        # Store models in heavy footprints
        for band, band_sources in self.band_sources.items():
            updateCatalogFootprints(
                modelData=model_data,
                catalog=band_sources,
                band=band,
                removeScarletData=False,
                updateFluxColumns=False,
            )

        print(type(self.orig.singles[0]))

        # for fast source lookup by id
        self.source_dict = {}
        for source in sources:
            source_id = source.getId()
            self.source_dict[source_id] = source

        self._set_footprints()
        self._build_heavies()
        self._build_subtracted_image()

    @contextmanager
    def add_source(self, source_id):
        """
        Open a with context that yields the image with all objects
        subtracted except the specified one.

        since the image had data-model this restores the pixels for the object
        of interest, minus models of other objects

        with subtractor.add_source(source_id):
            # do something with subtractor.mbexp

        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId()

        Yields
        -------
        ExposureF, although more typically one uses the .mbexp attribute
        """
        if source_id not in self.source_dict:
            raise ValueError(f'source {source_id} is not in the source list')

        self._add_or_subtract_source(source_id, 'add')
        # self._add_or_subtract_source_new(source_id, 'add')
        try:
            yield self.mbexp
        finally:
            self._add_or_subtract_source(source_id, 'subtract')
            # self._add_or_subtract_source_new(source_id, 'subtract')

    def _add_or_subtract_source(self, source_id, type):
        mbexp = self.mbexp
        scratch = self.scratch

        bbox = self.get_bbox(source_id)

        for band in self.bands:
            # Because footprints can only be used to *replace* pixels, we do so
            # on a scratch image and then subtract that from the model image

            heavy_fp = self.heavies[band][source_id]
            heavy_fp.insert(scratch[band].image)

            if type == 'add':
                mbexp[band].image[bbox] += scratch[band].image[bbox]
            else:
                mbexp[band].image[bbox] -= scratch[band].image[bbox]

            scratch[band].image[bbox] = 0

    def _add_or_subtract_source_new(self, source_id, type):
        mbexp = self.mbexp
        scratch = self.scratch

        bbox = self.get_bbox(source_id)

        for band in self.bands:
            # Because footprints can only be used to *replace* pixels, we do so
            # on a scratch image and then subtract that from the model image

            heavy_fp = self.heavies[band][source_id]
            heavy_fp.insert(scratch[band].image)

            if type == 'add':
                heavy_fp.addTo(mbexp[band].image[bbox])
            else:
                heavy_fp.subtractFrom(mbexp[band].image[bbox])

            scratch[band].image[bbox] = 0

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
                the source data because the model will have been added back in

            'original' means a stamp from the original data

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

        if source_id not in self.source_dict:
            raise ValueError(f'source {source_id} is not in the source list')

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
                the source data because the model will have been added back in

            'original' means a stamp from the original data

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

        if source_id not in self.source_dict:
            raise ValueError(f'source {source_id} is not in the source list')

        bbox = self.get_bbox(source_id, stamp_size=stamp_size, clip=clip)

        if type == 'original':
            mbexp = self.orig
        else:
            mbexp = self.mbexp

        source = self.source_dict[source_id]
        mbobs = ngmix.MultiBandObsList()

        for band in mbexp.bands:
            subexp = mbexp[band][bbox]
            obs = util.extract_obs(
                exp=subexp,
                source=source,
            )

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

        if source_id not in self.source_dict:
            raise ValueError(f'source {source_id} is not in the source list')

        scratch = self.scratch
        heavies = self.heavies

        bbox = self.get_bbox(source_id, stamp_size=stamp_size, clip=clip)

        exposures = []
        for band in self.bands:
            heavy_fp = heavies[band][source_id]
            heavy_fp.insert(scratch[band].image)

            model_exp = afw_image.ExposureF(scratch[band][bbox], deep=True)

            scratch[band].image[bbox] = 0

            exposures.append(model_exp)

        # return MultibandExposure.fromExposures(self.bands, exposures)
        return util.get_mbexp(exposures)

    def get_full_model(self):
        """
        Get a full model image of all sources.

        Returns
        -------
        ExposureF
        """
        heavies = self.heavies
        scratch = self.scratch

        model = util.copy_mbexp(self.mbexp, clear=True)

        for band, sources in self.band_sources.items():
            LOG.debug('-' * 70)
            LOG.debug(f'band: {band}')

            parents = sources.getChildren(0)
            for parent_record in parents:
                LOG.debug('parent id: %d', parent_record.getId())

                children = sources.getChildren(parent_record.getId())
                nchild = len(children)
                LOG.debug(
                    'processing %d %s',
                    nchild,
                    'children' if nchild > 1 else 'child',
                )

                for child in children:
                    child_id = child.getId()
                    heavy_fp = heavies[band][child_id]
                    heavy_fp.insert(scratch[band].image)

                    bbox = self.get_bbox(child_id)
                    model[band].image[bbox] += scratch[band].image[bbox]
                    scratch[band].image[bbox] = 0

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

        if source_id not in self.source_dict:
            raise ValueError(f'source {source_id} is not in the source list')

        # assumption: bounding boxes same in all bands
        band = self.bands[0]

        if stamp_size is not None:
            parent_id, fp = self.footprints[band][source_id]
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
            parent_id, fp = self.footprints[band][source_id]
            bbox = fp.getBBox()

        return bbox

    def _set_footprints(self):
        self.footprints = {}
        for band, sources in self.band_sources.items():
            self.footprints[band] = {
                source.getId(): (source.getParent(), source.getFootprint())
                for source in sources
            }

    def _build_heavies(self):
        self.heavies = {}
        for band, footprints in self.footprints.items():
            self.heavies[band] = {}

            for id, fp in footprints.items():
                # TODO test/ask if this is ok
                self.heavies[band][id] = afw_det.makeHeavyFootprint(
                    fp[1],
                    self.mbexp[band].maskedImage,
                )

                # if fp[1].isHeavy():
                #     self.heavies[band][id] = fp[1]
                # elif fp[0] == 0:
                #     self.heavies[band][id] = afw_det.makeHeavyFootprint(
                #         fp[1], self.mbexp[band].maskedImage,
                #     )

    def _build_subtracted_image(self):
        heavies = self.heavies
        mbexp = self.mbexp
        scratch = self.scratch

        for band, sources in self.band_sources.items():
            LOG.debug('-' * 70)
            LOG.debug(f'band: {band}')

            for source in sources:
                source_id = source.getId()
                heavy_fp = heavies[band][source_id]
                heavy_fp.insert(scratch[band].image)

                bbox = self.get_bbox(source_id=source_id)
                # mbexp[band].image[bbox] -= scratch[band].image[bbox]
                mbexp[band].image[bbox] -= scratch[band].image[bbox]
                scratch[band].image[bbox] = 0

    def _build_subtracted_image_old(self):
        heavies = self.heavies
        mbexp = self.mbexp
        scratch = self.scratch

        for band, sources in self.band_sources.items():
            LOG.debug('-' * 70)
            LOG.debug(f'band: {band}')

            parents = sources.getChildren(0)
            for parent_record in parents:
                LOG.debug('parent id: %d', parent_record.getId())

                children = sources.getChildren(parent_record.getId())
                nchild = len(children)
                LOG.debug(
                    'processing %d %s',
                    nchild,
                    'children' if nchild > 1 else 'child',
                )

                for child in children:
                    child_id = child.getId()
                    heavy_fp = heavies[band][child_id]
                    print('heavy_fp type:', type(heavy_fp))
                    heavy_fp.insert(scratch[band].image)

                    bbox = self.get_bbox(source_id=child_id)
                    # mbexp[band].image[bbox] -= scratch[band].image[bbox]
                    mbexp[band].image[bbox] -= scratch[band].image[bbox]
                    scratch[band].image[bbox] = 0
