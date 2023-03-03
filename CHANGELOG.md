# changelog

## unreleased

### added

### changed

### removed

### fixed

## 0.12.0 - 2023-04-03

### changed

 - Turned off coadding for joint fits by default.

## 0.11.0 - 2022-12-05

### changed

 - Gaussian fits are now done on coadds instead of a join fit across all shear bands.
 - Symmetrization for various fitters is now optional.
 - The Gaussian and adaptive moments fitters are now reused if possible.
 - Tolerances for lmfit are looser when doing Gaussian fits.

## 0.10.0 - 2022-12-01

### added

 - Added code to use adaptive moments and Gaussian fits for metadetect.


## 0.9.0 - 2022-08-11

### added

 - Added `fwhm_reg` parameter to regularize the moments.
 - Added support for more than one measurement made on the detections.
 - Added total image mask fraction to outputs.

### changed

 - Output data format column names have changed (`flags` -> `{model}_flags`, etc.)

### removed

 - Removed deconvolution of PSF with itself when smoothing.

## 0.8.1 - 2022-07-22

### fixed

 - Fixed bug where with smoothing the prePSF moments for PSFs need to be measured
   with the smoothing kernel applied.

## 0.8.0 - 2022-07-22

### added

 - Added adaptive moments to LSST metadetect.

### changed

- Changed code to detect ngmix versions and use the correct momments name (`mom` vs `sums`)
  for ngmix version less than 2.1 or greater than equal to 2.1.

### fixed

- Fixed bug where flux weighting was not applied to the PSF moments causing a weird
  stellar locus for weighted moments.

## 0.7.0 - 2022-05-03

### added

- Code can now make multiple shear measurements with different combinations of
  bands in a single pass.
- Code can now handle color-dependent PSF models.
- Code now keeps T flags.

### changed

- We no longer flag shears with |g| >= 1.
- Flags for band fluxes are now kept per band.


## 0.6.2 - 2022-01-21

### changed

 - DES metadetect now always fits the PSF.

### fixed

 - Fixed bug not converting x, y to local coordinate system applying apodized
   star mask for lsst
 - Fixed bug when creating an output structure for multiband lsst data when
   there was a measurement failure

## 0.6.1 - 2021-12-13

### fixed

 - Fixed bug where ngmix observations with all zero weights were not flagged
   properly.


## 0.6.0 - 2021-12-10

### changed

 - regular metadetect now defaults to using the sheared positions


## 0.5.0 - 2021-12-03

### added
 - Added edge apodization for LSST.
 - Added detected position quantities for DES codes.
 - Added tests for BRIGHT mask expansion in LSST.

### changed
 - LSST codes internally use DM exposures as opposed to ngmix observations.
 - All PSF fits are done only with adaptive moments.

### removed
 - Removed unused deblender code for LSST.
 - Removed DETECTED bit in LSST output catalogs.

### fixed
 - Fixed bug in LSST jacobians.


## 0.4.0 - 2021-11-17

### added
 - Weight masks now get 4-fold symmetry applied to zero-weight pixels before calling moments fitters.
 - Routines to do metacalibration directly with LSST exposure objects and galsim
 - Image edges are now apodized to reduce FFT artifacts for bright sources on the edges of images.
 - LSST codes now apodize circular masks applied to bright objects.
 - LSST codes now have a simple EM deblender for testing.
 - LSST codes now support multiple bands.

### changed
 - Shear values outside of |g| < 1 or |g1| < 1 or |g2| < 1 are now flagged in the fitters
 - The maskflags parameter was renamed to nodet_flags

### fixed
 - Scarlet's internal cache is now cleared to prevent memory usage issues


## 0.3.11 - 2021-10-20

### added
 - Added working implementations of LSST versions with scarlet.

### changed
 - Internal refactoring of moments measures. Now supports pgauss.

### removed
 - Dropped support for Gaussian fits.
 - Dropped support for ngmix 1.0


## 0.3.10 - 2021-10-02

### changed
 - Refactored LSST code to use functions instead of classes.

### added
 - Added new utils for interpolating in foreground mask holes.

### fixed
 - Fixed LSST deblending issues/usage.
 - Now return all measurements for LSST measurements for ksigma moments.
 - Various fixes in sky subtraction code including subtracting the sky on original images.


## 0.3.9 - 2021-06-25

### added
 - This release adds a step for lsst that determines and subtracts the sky


## 0.3.8 - 2021-06-25


## 0.3.7 - 2021-06-14

### added
 - Added mfrac measure for LSST metadetect codes

### fixed
 - Fixed a bug in the LSST deblend metadetect versions.


## 0.3.6 - 2021-06-07

### added
 - Added ksigma-moment support
 - Added tests for LSST versions


## 0.3.5 - 2021-05-11

### fixed
 - Fixed handling of edge cases where observations are completely masked or have all zero weights.


## 0.3.4 - 2021-05-08

### fixed
 - Fixed bad version bump with wrong version string.

## 0.3.3 - 2021-05-08

### fixed
 - Catch length error in LSST metadetect code.
 - Catch errors with missing bands or zero weights.


## 0.3.1 - 2021-05-06

### fixed
 - Fixed wrong class name for BaseLSSTMetadetect.
 - Fixed cases where gauss fitter tried to get fit quantities for objects with non-zero flags.


## 0.3.0 - 2021-05-06

### added
 - Code now supports ngmix v2.0
 - Added default of using sxdes for detection
 - Added flux measurements for non-shear bands
 - Added measurements of the masked fraction of each object from a masked fraction image
 - Added integration tests for shear recovery.

### changed
 - Moved CI to github actions

### removed
 - Removed old masked fraction columns


## 0.2.0 - 2021-02024

### added
 - Tagged version with license
