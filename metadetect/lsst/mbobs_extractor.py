from contextlib import contextmanager
from lsst.afw.image import MultibandExposure
from . import util


class MBObsExtractor(object):
    """
    A class to extract stamps around sources

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        A representation of the multi-band data set.
        Create one of these with
            descwl_model_subtractor.get_mbexp(exposure_list)
    sources: dict of lsst.afw.table.SourceCatalog
        This is the output of the ScarletDeblendTask, a dict of
        lsst.afw.table.SourceCatalog keyed by band

    Examples
    ---------

    subtractor = ModelSubtractor(exposure, sources)

    # add back one model; since the image had data-model this restores the
    # pixels for the object of interest, but with models of other objects
    # subtracted

    mbextractor = MBObsExtractor(mbexp, sources)

    mbobs = mbextractor.get_mbobs(source_id, stamp_size=stamp_size)

    with subtractor.add_source(source_id):
        # full MultibandExposure is in subtractor.mbexp

        stamp = subtractor.get_stamp(source_id, stamp_size=48)

    # model of the entire data set as a MultibandExposure
    full_model = subtractor.get_full_model()

    # model of one source
    model = subtractor.get_model(source_id, stamp_size=48)
    """

    def __init__(self, mbexp, sources):
        assert isinstance(mbexp, MultibandExposure)
        assert isinstance(sources, dict)
        self.mbexp = mbexp
        self.sources = sources

        self.source_ids = set()
        for source in sources[list(sources.keys())[0]]:
            self.source_ids.add(source.getId())

    @contextmanager
    def add_source(self, source_id):
        """
        Fake context manager so this class can be used as a stand-in
        for the ModelSubtractor


        Parameters
        ----------
        source_id: int
            The id of the source, e.g. from source.getId(), ignored
        """
        yield None

    def get_stamp(
        self,
        source_id,
        stamp_size,
        clip=False,
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
            A bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

        Returns
        -------
        ExposureF
        """

        if source_id not in self.source_ids:
            raise ValueError(f'source {source_id} is not in the source list')

        bbox = util.get_bbox(
            mbexp=self.mbexp,
            source=self.sources[source_id],
            stamp_size=stamp_size,
            clip=clip,
        )

        exposures = [self.mbexp[band][bbox] for band in self.filters]
        # return MultibandExposure.fromExposures(self.filters, exposures)
        return util.get_mbexp(exposures)

    def get_mbobs(
        self,
        source_id,
        stamp_size,
        clip=False,
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
            A bounding box is created with about this size rather than
            using the footprint bounding box. Typically the returned size is
            stamp_size + 1
        clip: bool, optional
            If set to True, clip the bbox to fit into the exposure.

            If clip is False and the bbox does not fit, a
            lsst.pex.exceptions.LengthError is raised

        Returns
        -------
        ngmix.MultiBandObsList
        """

        import ngmix

        if source_id not in self.source_ids:
            raise ValueError(f'source {source_id} is not in the source list')

        source = self.sources[source_id]

        bbox = util.get_bbox(
            mbexp=self.mbexp,
            source=source,
            stamp_size=stamp_size,
            clip=clip,
        )

        mbobs = ngmix.MultiBandObsList()

        for band in self.mbexp.bands:
            subexp = self.mbexp[band][bbox]
            obs = util.extract_obs(
                exp=subexp,
                source=source,
            )

            obslist = ngmix.ObsList(meta={'band': band})
            obslist.append(obs)
            mbobs.append(obslist)

        return mbobs
