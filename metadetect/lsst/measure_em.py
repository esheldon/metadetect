"""
I suspect the bias seen for shredder is due to the centers
not moving.  Implement simple em, one gauss per object, to
see if letting the centers move helps.  If it works we can
consider letting the center of each object in the shredder
move; this will take some coding
"""
import logging
import numpy as np
import ngmix

import lsst.afw.table as afw_table
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
)

from . import util
from .util import ContextNoiseReplacer
from . import vis
from .defaults import DEFAULT_THRESH
from ..fitting import get_wavg_output_struct
from .measure import AllZeroWeight, get_ormask, get_output
from ..procflags import ZERO_WEIGHTS

LOG = logging.getLogger('lsst_measure')


def measure(
    mbexp,
    rng,
    thresh=DEFAULT_THRESH,
    show=False,
):
    """
    run detection and deblending of peaks, as well as basic measurments such as
    centroid.  The SDSS deblender is run in order to split peaks.

    We must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures to process
    thresh: float, optional
        The detection threshold in units of the sky noise

    Returns
    -------
    sources, meas_task
        The sources and the measurement task
    """

    assert len(mbexp.singles) == 1, 'one band for now'

    if len(mbexp.singles) > 1:
        detexp = util.coadd_exposures(mbexp.singles)
    else:
        detexp = mbexp.singles[0]

    schema = afw_table.SourceTable.makeMinimalSchema()

    # Setup algorithms to run
    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        "base_SdssCentroid",
        "base_PsfFlux",
        "base_SkyCoord",
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

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    # variance here actually means relative to the sqrt(variance)
    # from the variance plane.
    # TODO this would include poisson
    # TODO detection doesn't work right when we tell it to trust
    # the variance
    # detection_config.thresholdType = 'variance'
    detection_config.thresholdValue = thresh

    detection_task = SourceDetectionTask(config=detection_config)

    # this must occur directly before any tasks are run because schema is
    # modified in place by tasks, and the constructor does a check that
    # fails if we construct it separately

    deblend_task = SourceDeblendTask(
        config=SourceDeblendConfig(),
        schema=schema,
    )

    table = afw_table.SourceTable.make(schema)

    result = detection_task.run(table, detexp)

    if show:
        vis.show_exp(detexp)

    if result is not None:
        res = []

        sources = result.sources
        deblend_task.run(detexp, sources)

        with ContextNoiseReplacer(detexp, sources, rng) as replacer:

            for source in sources:

                if source.get('deblend_nChild') != 0:
                    continue

                source_id = source.getId()

                with replacer.sourceInserted(source_id):
                    meas_task.callMeasure(source, detexp)
                    import IPython
                    IPython.embed()

            # now run through and put in parents
            parents = sources.getChildren(0)
            LOG.debug('found %d parents', len(parents))
            for parent in parents:

                with replacer.sourceInserted(parent.getId()):
                    tres = process_blend(rng, detexp, parent)
                    res += tres

    else:
        res = None

    return res


def process_blend(rng, exp, parent, ntry=2):
    from ngmix.gmix.gmix_nb import gmix_get_e1e2T
    from ngmix.gmix.gmix_nb import get_model_s2n_sum

    exp_bbox = exp.getBBox()
    wcs = exp.getWcs()

    if parent.get('deblend_nChild') != 0:
        sources = parent.getChildren()
    else:
        sources = [parent]

    try:
        obs = extract_obs_for_em(exp, parent)
    except AllZeroWeight as err:
        # failure creating some observation due to zero weights
        LOG.info('%s', err)
        return _make_multi_output(len(sources), flags=ZERO_WEIGHTS)

    # fit the psf
    psf_res = fit_psf(rng, obs)
    if psf_res['flags'] != 0:
        return _make_multi_output(len(sources), psf_flags=psf_res['flags'])

    # run em
    for i in range(ntry):
        guess = get_guess_em(rng=rng, obs=obs, sources=sources)

        res = ngmix.em.run_em(obs=obs, guess=guess)
        if res['flags'] == 0:
            break

    # fill in results
    if res['flags'] != 0:
        return _make_multi_output(
            len(sources), flags=res['flags'], psf_flags=psf_res['flags'],
        )
    else:
        gm = res.get_gmix()
        gmdata = gm.get_data()
        reslist = []
        for source in sources:
            res = _make_output(flags=0, psf_flags=0)

            this_gmdata = gmdata[i:i+1]
            e1, e2, T = gmix_get_e1e2T(this_gmdata)
            T_ratio = T/psf_res['T']

            s2n_sum = get_model_s2n_sum(this_gmdata, obs.pixels)

            if s2n_sum > 0:
                s2n = np.sqrt(s2n_sum)
            else:
                s2n = 0

            res['psf_g'] = psf_res['e']
            res['psf_T'] = psf_res['psf_T']
            res['g'] = (e1, e2)
            res['T'] = T
            res['s2n'] = s2n
            res['T_ratio'] = T_ratio

            ormask = get_ormask(source=source, exposure=exp)

            res = get_output(
                wcs=wcs,
                source=source,
                ormask=ormask,
                stamp_size=-1,
                exp_bbox=exp_bbox,
            )
            reslist.append(res)

    return reslist


def extract_obs_for_em(exp, parent):
    """
    convert an image object into an ngmix.Observation, including
    a psf observation

    parameters
    ----------
    imobj: lsst.afw.image.ExposureF
        The exposure
    parent: lsst.afw.table.SourceRecord
        The source record for the parent

    returns
    --------
    obs: ngmix.Observation
        The Observation unless all the weight are zero, in which
        case AllZeroWeight is raised
    """
    from .measure import (
        _extract_jacobian, _extract_weight, AllZeroWeight, extract_psf_image,
    )

    im = exp.image.array

    wt = _extract_weight(exp)
    if np.all(wt <= 0):
        raise AllZeroWeight('all weights <= 0')

    bmask = exp.mask.array
    jacob = _extract_jacobian(
        exp=exp,
        source=parent,
    )
    jacob.set_cen(0, 0)

    orig_cen = parent.getCentroid()

    psf_im = extract_psf_image(exposure=exp, orig_cen=orig_cen)

    # fake the psf pixel noise
    psf_err = psf_im.max()*0.0001
    psf_wt = psf_im*0 + 1.0/psf_err**2

    # use canonical center for the psf
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0
    psf_jacob = jacob.copy()
    psf_jacob.set_cen(row=psf_cen[0], col=psf_cen[1])

    # we will have need of the bit names which we can only
    # get from the mask object
    # this is sort of monkey patching, but I'm not sure of
    # a better solution

    meta = {'orig_cen': orig_cen}

    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacob,
    )
    obs = ngmix.Observation(
        im,
        weight=wt,
        bmask=bmask,
        jacobian=jacob,
        psf=psf_obs,
        meta=meta,
    )

    return obs


def fit_psf(rng, obs, ntry=4):
    Tguess0 = 0.25
    for i in range(ntry):
        Tguess = Tguess0 * rng.uniform(low=0.95, high=1.05)
        res = ngmix.admom.run_admom(
            obs.psf, guess=Tguess,
        )
        if res['flags'] == 0:
            break

    return res


def _make_output(flags=None, psf_flags=None):
    res = get_wavg_output_struct(nband=1, model='em')
    if flags is not None:
        res['flags'] = flags
    if psf_flags is not None:
        res['psf_flags'] = psf_flags
    return res


def _make_multi_output(num, flags=None, psf_flags=None):
    return [
        _make_output(flags=flags, psf_flags=psf_flags) for i in range(num)
    ]


def get_guess_em(
    rng,
    obs,
    sources,
    minflux=0.01,
    Tmin=0.23,  # not prepsf, but we could make it so
):
    """
    get a full gaussian mixture guess based on an input object list

    Parameters
    -----------
    rng: np.random.RandomState
        optional random number generator
    obs: ngmix.Observation
        The image data
    sources: sources: lsst.afw.table.SourceCatalog
        The sources in the blend
    minflux: float, optional
        Minimum flux allowed. Default 1.0

    Returns
    -------
    ngmix.GMix
    """

    ur = rng.uniform
    jacobian = obs.jacobian

    assert minflux > 0.0

    guess_pars = []

    for source in sources:
        # TODO proper guess for T
        Tguess = Tmin * ur(low=0.95, high=1.05)

        cen = source.getCentroid()
        row = cen.getY()
        col = cen.getX()

        if source.getPsfFluxFlag():
            flux = -9999
        else:
            flux = source.getPsfInstFlux()

        if flux < minflux:
            print('flux %g less than minflux %g' % (flux, minflux))
            flux = minflux * ur(low=0.95, high=1.05)

        v, u = jacobian.get_vu(row, col)

        g1, g2 = ur(low=-0.01, high=0.01, size=2)

        pars = [
            v,
            u,
            g1,
            g2,
            Tguess,
            flux,
        ]
        gm = ngmix.GMixModel(pars, 'gauss')

        guess_pars += list(gm.get_full_pars())

    gm_guess = ngmix.GMix(pars=guess_pars)

    return gm_guess
