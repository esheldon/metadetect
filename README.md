# metadetect

[![tests](https://github.com/lsst-dm/metadetect/actions/workflows/tests.yml/badge.svg)](https://github.com/lsst-dm/metadetect/actions/workflows/tests.yml)
[![shear-tests](https://github.com/lsst-dm/metadetect/actions/workflows/shear_test.yml/badge.svg)](https://github.com/lsst-dm/metadetect/actions/workflows/shear_test.yml)
[![lsst-tests](https://github.com/lsst-dm/metadetect/actions/workflows/lsst-tests.yml/badge.svg)](https://github.com/lsst-dm/metadetect/actions/workflows/lsst-tests.yml)

Library for meta-detection, combining detection and metacalibration.
The algorithm is explained in detail in [Sheldon et al., (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...902..138S/abstract) and its applicability with LSST data structures is demonstrated using simulations in [Sheldon et al., (2023)](https://ui.adsabs.harvard.edu/abs/2023OJAp....6E..17S/abstract).

## Shared-fork model

This repository is a fork of the original metadetection repository for packaging and distributing the `metadetect` code with, and for use within, LSST Science Pipelines (e.g., in `drp_tasks`).

### Motivation

We use a fork of this repository instead of declaring it as a dependency in `rubin-env` because the LSST-specific code in this repository uses some of the core packages of the LSST Science Pipelines themselves.
Having a fork allows us keeps the dependency graph cleaner and simpler, while enabling the Science Pipelines
team members to make any API changes in a consistent manner without breakage.
Any significant algorithmic change is expected to happen in the upstream package and merged into this fork.

The LSST-specific unit tests cannot be run on Jenkins because it has additional dependencies that are not available within `rubin-env` (e.g., `descwl-shear-sims`).
However, these tests are the most relevant ones for the LSST organization as they would indicate any breakage.
Therefore, we run these tests on GitHub Actions at least once weekly, using the latest weekly through `stackvana`.
While this does not help to catch breakage _before_ it is merged to the default branch, it helps identify it
soon after the change.
The workflow failures can be of two types: `ERRORS` typically due to incorrect APIs and `FAILURES` due to inaccurate results.
The latter may need to be fixed upstream after discussing with the original authors and are generally not within the scope of the LSST DM team.

### Style differences

This package differs from other LSST DM packages in the organization and coding styles.
Fixing these to adhere to the [LSST dev-guide](https://developer.lsst.io/python/style.html) is unnecessary code churn and makes it harder to pull in changes from the upstream.


- The directory organization of the package differs from typical LSST DM repository structure, but follows more of the community standard.
- The package uses `pytest` for unit tests instead of `unittest` package.
- Import statements need not be at the beginning of the module. On-the-fly imports are permitted so as to
not require having all the (optional) dependencies available.
- This LSST-specific code from this package is imported as `metadetect.lsst` as opposed to `lsst.metadetect`.
- The docstrings cannot be all parsed by `sphinx`, and cannot result in a clean, fully-built documentation.
