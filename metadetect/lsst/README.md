# Metadetect for Rubin Observatory / LSST

This subpackage contains code for running metadetect on data structures and algorithms available from the Rubin Science Pipelines.

## Config framework

### Overview of the config framework in Rubin Science Pipelines
The algorithms within Rubin Science Pipelines are implemented as `Task`s and `PipelineTask`s, each with an associate `Config` object.
This architecture allows using the same algorithm in multiple places, configured appropriately as required.
Thus, a `Config` object is initially mutable.
The default values are specified in the individual fields and they can be overridden in a special method called `setDefaults`.
However, once the `freeze` method is called, it becomes immutable (via public methods) and the values cannot be changed any further.
After a `Config` instance is frozen, its `validate` method is called to check that the configuration is valid one.
This is intended to check the data types and relationships between the fields.
When run as part of the pipelines, the pipeline execution (or `pex`) in Rubin Science Pipelines will call these methods before beginning the execution and persist them along with the processing runs for reproducibility.

### Use of Config framework in metadetect
The `metadetect` algorithm are also implemented as `Task`s with thin-wrappers around them for backwards compatibility.
The `Task` entry points are intended to be used in production, whereas the top-level wrapper functions continue to serve as entry points for on-the-fly simulations with other packages (e.g., [descwl-shear-sims](https://github.com/LSSTDESC/descwl-shear-sims) and [descwl_coadd](https://github.com/LSSTDESC/descwl_coadd)).
In order to prevent changes to the default values upstream going unnoticed and affecting the performance of `metadetect`, the unit tests in `tests/test_lsst_dm_configs.py` check that the `Config` objects what they have been validated on simulations.
