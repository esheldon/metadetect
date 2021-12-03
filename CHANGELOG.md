# changelog

## unreleased

### added

### changed

### removed

### fixed


## 0.5.0 - 2021-12-02

### added
 - Added edge apodization for LSST.
 - Added detected position quantities for DES codes.
 - Added tests for BRIGHT mask expansion in LSST.

### changed
 - LSST codes internally use DM exposures as opposed to ngmix observations.
 - All PSF fits are done only with adaptive moments.

### removed
 - Removed unused deblender code for LSST.
 - Removed detected bit in LSST output catalogs.

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
