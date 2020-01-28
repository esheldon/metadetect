import lsst.afw.table as afw_table
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
    NoiseReplacerConfig,
    NoiseReplacer,
)
import lsst.log

from .detect import MEDSifier
from .lsst_mbobs_extractor import MBObsExtractor


class LSSTMEDSifier(MEDSifier):
    def __init__(self, *, mbobs, meds_config, psf_fwhm_arcsec):
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.psf_fwhm_arcsec = psf_fwhm_arcsec

        assert len(mbobs) == 1, 'multi band not supported yet'
        assert len(mbobs[0]) == 1, 'multi-epoch is not supported'

        self._set_meds_config(meds_config)
        self.log = lsst.log.Log.getLogger("LSSTMEDSifier")

        self.log.debug('setting detection exposure')
        self._set_detim_exposure()

        self.log.debug('detecting and deblending')
        self._detect_and_deblend()

        self.log.debug('setting exposure and psf lists')
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

    def show(self):
        """
        show the detection image
        """
        import matplotlib.pyplot as plt
        plt.imshow(
            self.mbobs[0][0].image,
            interpolation='nearest',
            cmap='gray',
        )
        plt.show()

    def _detect_and_deblend(self):
        exposure = self.det_exp

        # This schema holds all the measurements that will be run within the
        # stack It needs to be constructed before running anything and passed
        # to algorithms that make additional measurents.
        schema = afw_table.SourceTable.makeMinimalSchema()

        # Setup algorithms to run
        meas_config = SingleFrameMeasurementConfig()
        meas_config.plugins.names = [
            "base_SdssCentroid",
            "base_PsfFlux",
            "base_SkyCoord",
            # "base_SdssShape",
            # "base_LocalBackground",
        ]

        # set these slots to none because we aren't running these algorithms
        meas_config.slots.apFlux = None
        meas_config.slots.gaussianFlux = None
        meas_config.slots.calibFlux = None
        meas_config.slots.modelFlux = None

        # goes with SdssShape above
        meas_config.slots.shape = None

        # fix odd issue where it things things are near the edge
        meas_config.plugins['base_SdssCentroid'].binmax = 1

        meas_task = SingleFrameMeasurementTask(
            config=meas_config,
            schema=schema,
        )

        # setup detection config
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

        # Run on deblended images
        noise_replacer_config = NoiseReplacerConfig()
        footprints = {
            record.getId(): (record.getParent(), record.getFootprint())
            for record in result.sources
        }

        # This constructor will replace all detected pixels with noise in the
        # image
        replacer = NoiseReplacer(
            noise_replacer_config,
            exposure=exposure,
            footprints=footprints,
        )

        nbad = 0
        ntry = 0
        kept_sources = []

        for record in result.sources:

            # Skip parent objects where all children are inserted
            if record.get('deblend_nChild') != 0:
                continue

            ntry += 1

            # This will insert a single source into the image
            replacer.insertSource(record.getId())    # Get the peak as before

            # peak = record.getFootprint().getPeaks()[0]

            # The bounding box will be for the parent object
            # bbox = record.getFootprint().getBBox()

            meas_task.callMeasure(record, exposure)

            # Remove object
            replacer.removeSource(record.getId())

            if record.getCentroidFlag():
                nbad += 1

            kept_sources.append(record)

        # Insert all objects back into image
        replacer.end()

        if ntry > 0:
            self.log.debug('nbad center: %d frac: %d' % (nbad, nbad/ntry))

        nkeep = len(kept_sources)
        ntot = len(result.sources)
        self.log.debug('kept %d/%d non parents' % (nkeep, ntot))
        self.sources = kept_sources
