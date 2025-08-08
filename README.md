# metadetect


Library for meta-detection, combining detection and metacalibration.
The algorithm is explained in detail in [Sheldon et al., (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...902..138S/abstract) and its applicability with LSST data structures is demonstrated using simulations in [Sheldon et al., (2023)](https://ui.adsabs.harvard.edu/abs/2023OJAp....6E..17S/abstract).

## Shared-fork model

This repository is a fork of the original metadetection repository for packaging and distributing the `metadetect` code with, and for use within, LSST Science Pipelines (e.g., in `drp_tasks`).

### Motivation

We use a fork of this repository instead of declaring it as a dependency in `rubin-env` because the LSST-specific code in this repository uses some of the core packages of the LSST Science Pipelines themselves.
Having a fork allows us keeps the dependency graph cleaner and simpler, while enabling the Science Pipelines
team members to make any API changes in a consistent manner without breakage.
Typically, any significant algorithmic change is expected to first happen in the upstream package and then merged into this fork.
However, because of the package's dependency on the LSST Science Pipelines, some changes may need to be merge to `lsst-dev` first before being ported upstream.

The LSST-specific unit tests are run on Jenkins (`stack-os-matrix` and `nightly-release` etc.) but it has additional dependencies that are not distributed as part of `lsst_distrib` (e.g., `descwl-shear-sims`, `descwl_coadd`).
These are DESC packages but we rely on forks within the `lsst-dm` organization.
They should be synced regularly after auditing, say, for new dependencies that may not be in `rubin-env`, API changes that break the tests etc.
The test failures on the CI system can be of two types: `ERRORS` typically due to incorrect APIs and `FAILURES` due to inaccurate results.
The latter needs to be discussed in dm-science-pipelines channel in LSST-DA or dm-algorithms-pipelines in the Rubin Observatory Slack workspaces.
The `FAILURES` may be temporarily resolved by relaxing the thresholds, but it should promptly be fixed upstream after consulting the original authors.

### Style differences

This package differs from other LSST DM packages in the organization and coding styles.
Fixing these to adhere to the [LSST dev-guide](https://developer.lsst.io/python/style.html) is unnecessary code churn and makes it harder to pull in changes from the upstream.


- The directory organization of the package differs from typical LSST DM repository structure, but follows more of the community standard.
- The package uses `pytest` for unit tests instead of `unittest` package.
- Import statements need not be at the beginning of the module. On-the-fly imports are permitted so as to
not require having all the (optional) dependencies available.
- This LSST-specific code from this package is imported as `metadetect.lsst` as opposed to `lsst.metadetect`.
- The docstrings cannot be all parsed by `sphinx`, and cannot result in a clean, fully-built documentation.
