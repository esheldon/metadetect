import logging
import lsst.afw.table as afw_table
import lsst.afw.image as afw_image
import lsst.geom as geom
from lsst.afw.geom import makeSkyWcs, makeCdMatrix
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig

from .detect import MEDSifier
from .lsst_mbobs_extractor import MBObsExtractor

LOGGER = logging.getLogger(__name__)


class LSSTMEDSifier(MEDSifier):
    def __init__(self, *, mbobs, meds_config, psf_fwhm_arcsec):
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.psf_fwhm_arcsec = psf_fwhm_arcsec

        assert len(mbobs) == 1, 'multi band not supported yet'
        assert len(mbobs[0]) == 1, 'multi-epoch is not supported'

        self._set_meds_config(meds_config)

        # LOGGER.info('setting detection image')
        # self._set_detim()

        LOGGER.info('setting detection exposure')
        self._set_detim_exposure()

        LOGGER.info('detecting and deblending')
        self._detect_and_deblend()

        LOGGER.info('setting exposure and psf lists')
        self._set_exposures_and_psfs()

    def get_multiband_meds(self):
        config = {
            'stamps': {
                'min_stamp_size': 32,
                'max_stamp_size': 128,
                'sigma_factor': 5,
                'bits_to_ignore_for_weight': [],
                'bits_to_null': [],
            }
        }
        return MBObsExtractor(
            config=config,
            exposures=self.exposures,
            sources=self.sources,
            psfs=self.psfs,
        )

    def _set_exposures_and_psfs(self):
        self.psfs = None
        self.exposures = [self.mbobs[0][0].exposure]

    def _set_detim_exposure(self):
        """
        in this one we set a gaussian psf for detection
        """
        self.det_exp = self.mbobs[0][0].exposure

    def _detect_and_deblend(self):
        exposure = self.det_exp

        # This schema holds all the measurements that will be run within the
        # stack It needs to be constructed before running anything and passed
        # to algorithms that make additional measurents.
        schema = afw_table.SourceTable.makeMinimalSchema()

        detection_config = SourceDetectionConfig()
        detection_config.reEstimateBackground = False
        detection_config.thresholdValue = 10
        detection_task = SourceDetectionTask(config=detection_config)

        deblend_config = SourceDeblendConfig()
        deblend_task = SourceDeblendTask(config=deblend_config, schema=schema)

        # Detect objects
        table = afw_table.SourceTable.make(schema)
        result = detection_task.run(table, exposure)
        sources = result.sources

        # run the deblender
        deblend_task.run(exposure, sources)

        self.sources = sources


def get_exposure_from_obs(obs):
    ny, nx = obs.image.shape

    scale = obs.jacobian.scale
    masked_image = afw_image.MaskedImageF(nx, ny)
    masked_image.image.array[:] = obs.image

    # assuming constant noise
    var = 1.0/obs.weight[0, 0]
    masked_image.variance.array[:] = var
    masked_image.mask.array[:] = 0

    exp = afw_image.ExposureF(masked_image)

    # set WCS
    orientation = 0*geom.degrees

    cd_matrix = makeCdMatrix(
        scale=scale*geom.arcseconds,
        orientation=orientation,
    )
    crpix = geom.Point2D(nx/2, ny/2)
    crval = geom.SpherePoint(0, 0, geom.degrees)
    wcs = makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cd_matrix)
    exp.setWcs(wcs)

    return exp
