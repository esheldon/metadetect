import logging
from contextlib import contextmanager, ExitStack
import ngmix
import numpy as np

logger = logging.getLogger(__name__)


class ContextNoiseReplacer(object):
    """
    noise replacer that works as a context manager

    Parameters
    ----------
    mbexp: lsst.afw.image.Exposure
        The data
    sources: lsst.afw.table.SourceCatalog
        Catalog of sources
    noise_image: array, optional
        Optional noise image to use.  If not sent one is generated.

    Examples
    --------
    with ContextNoiseReplacer(exposure=exp, sources=sources) as replacer:
        # do something
    """

    def __init__(self, exposure, sources, rng, noise_image=None):
        from lsst.meas.base import NoiseReplacerConfig, NoiseReplacer

        # Notes for metacal.
        #
        # For metacal we should generate a noise image so that the exact noise
        # field is used for all versions of the metacal images.  The assumption is
        # that, because these noise data should contain no signal, metacal is not
        # calibrating it.  Thus it doesn't matter whether or not the noise field is
        # representative of the full covariance of the true image noise.  Rather by
        # making the field the same for all metacal images we reduce variance in
        # the calculation of the response

        config = NoiseReplacerConfig()

        # TODO DM needs to fix the crash
        # config.noiseSource = 'variance'
        config.noiseSeedMultiplier = rng.randint(0, 2**24)

        if noise_image is None:
            # TODO remove_poisson should be true for real data
            noise_image = get_noise_image(exposure, rng=rng, remove_poisson=False)

        footprints = {
            source.getId(): (source.getParent(), source.getFootprint())
            for source in sources
        }

        # This constructor will replace all detected pixels with noise in the
        # image
        self.replacer = NoiseReplacer(
            config,
            exposure=exposure,
            footprints=footprints,
            noiseImage=noise_image,
        )

    @contextmanager
    def sourceInserted(self, source_id):
        """
        Context manager to insert a source

        with replacer.sourceInserted(source_id):
            # do something with exposure
        """
        self.insertSource(source_id)

        try:
            yield self
        finally:
            self.removeSource(source_id)

    def insertSource(self, source_id):
        """
        Insert a source
        """
        self.replacer.insertSource(source_id)

    def removeSource(self, source_id):
        """
        Remove a source
        """
        self.replacer.removeSource(source_id)

    def end(self):
        self.replacer.end()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.end()


class MultibandNoiseReplacer(object):
    """
    noise replacer that works on multiple bands

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The data
    sources: lsst.afw.table.SourceCatalog
        Catalog of sources

    Examples
    --------
    with MultibandNoiseReplacer(mbexp=mbexp, sources=sources) as replacer:
        with replacer.insertSource(source_id):
            # do something with exposures
    """
    def __init__(self, mbexp, sources, rng):
        self.mbexp = mbexp
        self.sources = sources
        self.rng = rng
        self._set_noise_replacers()

    @contextmanager
    def sourceInserted(self, source_id):
        """
        Context manager to insert a source

        with replacer.sourceInserted(source_id):
            # do something with exposures
        """
        self.insertSource(source_id)

        try:
            # usually won't use this yielded value
            yield self.mbexp
        finally:
            self.removeSource(source_id)

    def insertSource(self, source_id):
        """
        Insert a source
        """
        for replacer in self.noise_replacers:
            replacer.insertSource(source_id)

    def removeSource(self, source_id):
        """
        Remove a source
        """
        for replacer in self.noise_replacers:
            replacer.removeSource(source_id)

    def end(self):
        """
        end the noise replacment.  Called automatically upon leaving the
        context manager
        """
        self.exit_stack.close()

    def _set_noise_replacers(self):
        """
        set a noise replacer for each exp and put it on the
        context lib ExitStack for cleanup later
        """

        self.noise_replacers = []
        self.exit_stack = ExitStack()

        for exp in self.mbexp.singles:
            replacer = ContextNoiseReplacer(
                exposure=exp,
                sources=self.sources,
                rng=self.rng,
            )
            self.noise_replacers.append(replacer)
            self.exit_stack.enter_context(replacer)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.end()


def get_noise_replacer(exposure, sources, noise_image=None):
    """
    get a regular noise replacer for the input exposure and source list

    Parameters
    ----------
    exposure: lsst.afw.image.ExposureF
        The exposure data
    sources: lsst.afw.table.SourceCatalog
        The sources
    noise_image: array, optional
        Optional array of noise

    Returns
    -------
    lsst.meas.base.NoiseReplacer
    """
    from lsst.meas.base import NoiseReplacerConfig, NoiseReplacer

    # Notes for metacal.
    #
    # For metacal we should generate a noise image so that the exact noise
    # field is used for all versions of the metacal images.  The assumption is
    # that, because these noise data should contain no signal, metacal is not
    # calibrating it.  Thus it doesn't matter whether or not the noise field is
    # representative of the full covariance of the true image noise.  Rather by
    # making the field the same for all metacal images we reduce variance in
    # the calculation of the response

    noise_replacer_config = NoiseReplacerConfig()
    footprints = {
        source.getId(): (source.getParent(), source.getFootprint())
        for source in sources
    }

    # This constructor will replace all detected pixels with noise in the
    # image
    return NoiseReplacer(
        noise_replacer_config,
        exposure=exposure,
        footprints=footprints,
        noiseImage=noise_image,
    )


def get_noise_image(exp, rng, remove_poisson):
    """
    get a noise image based on the input exposure

    TODO gain correct separately in each amplifier, currently
    averaged

    Parameters
    ----------
    exp: afw.image.ExposureF
        The exposure upon which to base the noise
    rng: np.random.RandomState
        The random number generator for making the noise image
    remove_poisson: bool
        If True, remove the poisson noise from the variance
        estimate.

    Returns
    -------
    MaskedImage
    """
    import lsst.afw.image as afw_image

    noise_exp = afw_image.ExposureF(exp, deep=True)

    signal = exp.image.array
    variance = exp.variance.array

    use = np.where(np.isfinite(variance) & np.isfinite(signal))

    if remove_poisson:
        gains = [
            amp.getGain() for amp in exp.getDetector().getAmplifiers()
        ]
        mean_gain = np.mean(gains)

        corrected_var = variance[use] - signal[use] / mean_gain

        var = np.median(corrected_var)
    else:
        var = np.median(variance[use])

    noise = rng.normal(scale=np.sqrt(var), size=signal.shape)

    noise_exp.image.array[:, :] = noise
    noise_exp.variance.array[:, :] = var

    return noise_exp.getMaskedImage()


def get_mbexp(exposures):
    """
    convert a list of exposures into an MultibandExposure

    Parameters
    ----------
    exposures: [lsst.afw.image.ExposureF]
        List of exposures of type F, D etc.

    Returns
    -------
    lsst.afw.image.MultibandExposure
    """
    from lsst.afw.image import MultibandExposure

    filters = [exp.getFilterLabel().bandLabel for exp in exposures]
    mbexp = MultibandExposure.fromExposures(filters, exposures)

    for exp, sexp in zip(exposures, mbexp.singles):
        sexp.setFilterLabel(exp.getFilterLabel())
        sexp.setWcs(exp.getWcs())

    return mbexp


def copy_mbexp(mbexp, clear=False):
    """
    copy a MultibandExposure with psfs

    clone() does not copy the psfs

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to copy
    clear: bool, optional
        If set to True, clear the data

    Returns
    -------
    lsst.afw.image.MultibandExposure
    """

    new_mbexp = mbexp.clone()

    # clone does not copy the psfs
    for band in mbexp.filters:
        psf = try_clone_psf(mbexp[band].getPsf())
        new_mbexp[band].setPsf(psf)

    for new_exp, exp in zip(new_mbexp.singles, mbexp.singles):
        new_exp.setFilterLabel(exp.getFilterLabel())
        new_exp.setWcs(exp.getWcs())

    if clear:
        new_mbexp.image.array[:, :, :] = 0

    return new_mbexp


def try_clone_psf(psf):
    """
    try to clone it, if not return a reference
    """
    try:
        new_psf = psf.clone()
    except RuntimeError:
        # this happens on some psfs, not sure why
        # proceed without a copy
        new_psf = psf

    return new_psf


def coadd_exposures(exposures):
    """
    coadd a set of exposures, assuming they share the same wcs

    Parameters
    ----------
    exposures: [lsst.afw.image.Exposure]
        List of exposures to coadd

    Returns
    --------
    lsst.afw.image.ExposureF
    """
    import lsst.geom as geom
    import lsst.afw.image as afw_image
    from lsst.meas.algorithms import KernelPsf
    from lsst.afw.math import FixedKernel

    wsum = 0.0

    for i, exp in enumerate(exposures):

        shape = exp.image.array.shape

        ycen, xcen = (np.array(shape) - 1)/2
        cen = geom.Point2D(xcen, ycen)

        psfobj = exp.getPsf()
        this_psfim = psfobj.computeKernelImage(cen).array

        if i == 0:
            coadd_exp = afw_image.ExposureF(exp, deep=True)
            coadd_exp.image.array[:, :] = 0.0

            weight = np.zeros(shape, dtype='f4')

            psf_im = this_psfim.copy()
            psf_im[:, :] = 0

        coadd_exp.mask.array |= exp.mask.array

        w = np.where(exp.variance.array > 0)
        medvar = np.median(exp.variance.array[w])
        this_weight = 1.0/medvar
        # print('medvar', medvar)

        coadd_exp.image.array[w] += exp.image.array[w] * this_weight
        psf_im += this_psfim * this_weight

        weight[w] += 1.0/exp.variance.array[w]

        wsum += this_weight

    fac = 1.0/wsum

    coadd_exp.image.array[:, :] *= fac

    # the psf is always normalized
    psf_im *= 1.0/psf_im.sum()

    coadd_exp.variance.array[:, :] = np.inf
    w = np.where(weight > 0)
    coadd_exp.variance.array[w] = 1/weight[w]

    coadd_psf = KernelPsf(FixedKernel(afw_image.ImageD(psf_im)))
    coadd_exp.setPsf(coadd_psf)

    coadd_exp.setWcs(exposures[0].getWcs())

    return coadd_exp


def coadd_mbobs(mbobs):
    """
    coadd a set of exposures, assuming they share the same wcs

    Parameters
    ----------
    exposures: [lsst.afw.image.Exposure]
        List of exposures to coadd

    Returns
    --------
    lsst.afw.image.ExposureF
    """

    wsum = 0.0

    meta = {}
    for i, obslist in enumerate(mbobs):
        obs = obslist[0]
        meta.update(obs.meta)

        shape = obs.image.shape

        if i == 0:
            coadd_image = np.zeros(shape, dtype='f4')
            coadd_weight = np.zeros(shape, dtype='f4')

            coadd_psf = np.zeros(obs.psf.image.shape)

        w = np.where(obs.weight > 0)
        this_weight = np.median(obs.weight[w])

        coadd_image[:, :] += obs.image * this_weight
        coadd_weight[:, :] += obs.weight
        coadd_psf[:, :] += obs.psf.image * this_weight

        wsum += this_weight

    fac = 1.0/wsum

    coadd_image[:, :] *= fac

    # the psf is always normalized
    coadd_psf *= 1.0/coadd_psf.sum()

    # use the jacobians from the last obs
    jac = obs.jacobian
    psf_jac = obs.psf.jacobian

    fake_noise = coadd_psf.max() / 1.e6
    psf_weight = np.zeros(coadd_psf.shape)
    psf_weight += (1.0/fake_noise**2)

    psf_obs = ngmix.Observation(
        image=coadd_psf,
        weight=psf_weight,
        jacobian=psf_jac,
    )
    output_obs = ngmix.Observation(
        image=coadd_image,
        weight=coadd_weight,
        jacobian=jac,
        psf=psf_obs,
        meta=meta,
    )

    return output_obs


def trim_odd_image(im):
    """
    trim an odd dimension image to by square and with equal distance from
    canonical center to all edges
    """

    dims = im.shape
    if dims[0] != dims[1]:
        logger.debug('original dims: %s' % str(dims))
        assert dims[0] % 2 != 0, 'image must have odd dims'
        assert dims[1] % 2 != 0, 'image must have odd dims'

        dims = np.array(dims)
        cen = (dims-1)//2
        cen = cen.astype('i4')

        distances = (
            cen[0]-0,
            dims[0]-cen[0]-1,
            cen[1]-0,
            dims[1]-cen[1]-1,
        )
        logger.debug('distances: %s' % str(distances))
        min_dist = min(distances)

        start_row = cen[0] - min_dist
        end_row = cen[0] + min_dist
        start_col = cen[1] - min_dist
        end_col = cen[1] + min_dist

        # adding +1 for slices
        new_im = im[
            start_row:end_row+1,
            start_col:end_col+1,
        ].copy()

        logger.debug('new dims: %s' % str(new_im.shape))

    else:
        new_im = im

    return new_im


def exp2obs(exp):
    """
    convert an exposure to an observation.

    The origin for the jacobian is set to the image center. The
    psf image is reconstructed at the center of the image.

    Parameters
    ----------
    exp: lsst.afw.image.ExposureF
        The exposure

    Returns
    -------
    ngmix.Observation
    """
    import lsst.geom as geom
    dims = exp.image.array.shape

    row, col = (np.array(dims)-1)/2
    cen = geom.Point2D(x=col, y=row)

    wcs = exp.getWcs()
    dm_jac = wcs.linearizePixelToSky(cen, geom.arcseconds)
    matrix = dm_jac.getLinear().getMatrix()
    jac = ngmix.Jacobian(
        x=col,
        y=row,
        dudx=matrix[1, 1],
        dudy=-matrix[1, 0],
        dvdx=matrix[0, 1],
        dvdy=-matrix[0, 0],
    )

    weight = np.zeros(dims)
    w = np.where(exp.variance.array > 0)
    weight[w] = 1.0/exp.variance.array[w]

    psf_obj = exp.getPsf()
    psf_image = psf_obj.computeKernelImage(cen).array

    psf_cen = (np.array(psf_image.shape)-1.0)/2.0

    psf_jac = jac.copy()
    psf_jac.set_cen(row=psf_cen[0], col=psf_cen[1])

    psf_err = psf_image.max()*0.0001
    psf_weight = psf_image*0 + 1.0/psf_err**2

    psf_obs = ngmix.Observation(
        image=psf_image,
        weight=psf_weight,
        jacobian=psf_jac,
    )

    return ngmix.Observation(
        image=exp.image.array,
        weight=weight,
        bmask=exp.mask.array,
        jacobian=jac,
        psf=psf_obs,
    )
