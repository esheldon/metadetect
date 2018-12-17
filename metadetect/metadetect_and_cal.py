import copy
import ngmix
import numpy as np
import esutil as eu
import galsim

from . import detect
from . import fitting

STEP = 0.01
SHEARS = {
    '1p': galsim.Shear(g1=STEP, g2=0.0),
    '1m': galsim.Shear(g1=-STEP, g2=0.0),
    '2p': galsim.Shear(g1=0.0, g2=STEP),
    '2m': galsim.Shear(g1=0.0, g2=-STEP)}


def do_metadetect_and_cal(
        config, mbobs, rng, wcs_jacobian_func=None, psf_rec_funcs=None):
    """
    Meta-detect and cal on the multi-band observations.
    """
    md = MetadetectAndCal(
        config, mbobs, rng,
        wcs_jacobian_func=wcs_jacobian_func, psf_rec_funcs=psf_rec_funcs)
    md.go()
    return md.result


class MetadetectAndCal(dict):
    """
    Meta-detect and cal on the multi-band observations.

    parameters
    ----------
    config: dict
        Configuration dictionary. Possible entries are
            metacal
            weight (if calculating weighted moments)
            max: (if running a max like fitter)
            fofs (if running MOF)
            mof (if running MOF)

    """
    def __init__(
            self, config, mbobs, rng, wcs_jacobian_func=None,
            psf_rec_funcs=None):
        self._set_config(config)
        self.mbobs = mbobs
        self.nband = len(mbobs)
        self.rng = rng
        self.wcs_jacobian_func = wcs_jacobian_func
        if psf_rec_funcs is None:
            self.psf_rec_funcs = [None] * len(mbobs)
        else:
            assert len(psf_rec_funcs) == len(mbobs)
            self.psf_rec_funcs = psf_rec_funcs

        self._set_fitter()

    @property
    def result(self):
        """
        get the result dict, keyed by the metacal type such
        as 'noshear', '1p', '1m', '2p', '2m'
        """
        if not hasattr(self, '_result'):
            raise RuntimeError('run go() first')

        return self._result

    def go(self):
        """
        make sheared versions of the images, run detection and measurements
        on each
        """
        odict = self._get_all_metacal()

        self._result = {}
        for key, sheared_mbobs in odict.items():
            self._result[key] = self._measure(sheared_mbobs, key)

    def _measure(self, sheared_mbobs, mcal_step):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements
        """

        if mcal_step == 'noshear':
            pos_transform = None
        else:
            # here we undo the shearing operation
            # the steps are
            # 1. take (x, y) to (u, v) using the ngmix jacobian
            # 2. undo the shear about the canonical image in (u, v)
            # 3. take the unsheared (u, v) back to (x, y)

            # this is the matrix that undoes shearing in u, v
            ainv = np.linalg.inv(SHEARS[mcal_step].getMatrix())

            # this is the jacobian that deals with the WCS
            jac = self.mbobs[0][0].jacobian.copy()

            # we need the canonical image center in (u, v) for undoing the
            # shearing
            dims = sheared_mbobs[0][0].image.shape
            x_cen = (dims[0] - 1) / 2
            y_cen = (dims[1] - 1) / 2
            v_cen, u_cen = jac.get_vu(row=y_cen, col=x_cen)

            def pos_transform(x, y):
                # apply WCS to get to world coords
                v, u = jac.get_vu(row=y, col=x)

                # unshear (subtract and then add the canonical center in u, v)
                u = np.atleast_1d(u) - u_cen
                v = np.atleast_1d(v) - v_cen
                out = np.dot(ainv, np.vstack([u, v]))
                assert out.shape[1] == u.shape[0]
                u = out[0] + u_cen
                v = out[1] + v_cen

                # undo the WCS to get to image coords
                ynew, xnew = jac.get_rowcol(v=v, u=u)

                return xnew, ynew

        # returns a MultiBandNGMixMEDS interface for the sheared positions
        # on the **original** image
        mbm, cat = self._do_detect(
            sheared_mbobs, pos_transform_func=pos_transform)
        mbobs_list = mbm.get_mbobs_list()

        # do the desired mcal step
        mcal_config = copy.deepcopy(self['metacal'])
        mcal_config['force_required_types'] = False
        mcal_config['types'] = [mcal_step]
        mcal_mbobs_list = []
        for mbobs in mbobs_list:
            mcal_dict = ngmix.metacal.get_all_metacal(
                mbobs,
                rng=self.rng,
                step=STEP,
                **mcal_config,
            )
            mcal_mbobs_list.append(mcal_dict[mcal_step])

        res = self._fitter.go(mbobs_list)

        res = self._add_positions(cat, res)
        return res

    def _set_fitter(self):
        """
        set the fitter to be used
        """
        self._fitter = fitting.Moments(
            self,
            self.rng,
        )

    def _add_positions(self, cat, res):
        """
        add catalog positions to the result
        """
        if cat.size > 0:
            new_dt = [
                ('sx_row', 'f4'),
                ('sx_col', 'f4'),
            ]
            newres = eu.numpy_util.add_fields(
                res,
                new_dt,
            )

            newres['sx_col'] = cat['x']
            newres['sx_row'] = cat['y']
        else:
            newres = res

        return newres

    def _do_detect(self, sheared_mbobs, pos_transform_func=None):
        """
        use a MEDSifier to run detection
        """
        sheared_mer = detect.MEDSifier(
            sheared_mbobs,
            sx_config=self['sx'],
            meds_config=self['meds'],
            wcs_jacobian_func=self.wcs_jacobian_func,
            pos_transform_func=pos_transform_func,
        )

        # now build the meds interface on the **orig** obs
        mlist = []
        for obslist, psf_rec in zip(self.mbobs.copy(), self.psf_rec_funcs):
            obs = obslist[0]
            mlist.append(detect.MEDSInterface(
                obs,
                sheared_mer.seg,
                sheared_mer.cat,
                psf_rec_func=psf_rec))

        return detect.MultiBandNGMixMEDS(mlist), sheared_mer.cat

    def _get_all_metacal(self):
        """
        get the sheared versions of the observations
        """

        if self['metacal'].get('symmetrize_psf', False):
            assert 'psf' in self, 'need psf fitting for symmetrize_psf'
            fitting.fit_all_psfs(self.mbobs, self['psf'], self.rng)

        odict = ngmix.metacal.get_all_metacal(
            self.mbobs,
            rng=self.rng,
            step=STEP,
            **self['metacal']
        )

        return odict

    def _set_config(self, config):
        """
        set the config, dealing with defaults
        """

        self.update(config)
        assert 'metacal' in self, \
            'metacal setting must be present in config'
        assert 'sx' in self, \
            'sx setting must be present in config'
        assert 'meds' in self, \
            'meds setting must be present in config'
