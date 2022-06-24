"""
test configs

note we allow the get_all_metacal to do its own verifiction
of the metacal sub config
"""
import pytest

from metadetect.lsst.configs import get_config
from metadetect.lsst.defaults import DEFAULT_FWHM_SMOOTH


def test_configs_smoke():
    config = get_config()

    # make sure the default is verified
    get_config(config)

    with pytest.raises(ValueError):
        get_config({'blah': 3})


@pytest.mark.parametrize('meas_type', ['am', 'wmom', 'gauss', 'coellip3'])
def get_weight_config(meas_type):
    inconfig = {}

    # make sure the default is verified
    get_config(inconfig)

    inconfig = {
        'meas_type': meas_type,
    }
    get_config(inconfig)

    fwhm = 1.2
    fwhm_smooth = 0.8
    for wtc in [{'fwhm': fwhm}, {'fwhm': fwhm, 'fwhm_smooth': fwhm_smooth}]:
        inconfig = {
            'meas_type': meas_type,
            'weight': wtc,
        }
        config = get_config(inconfig)
        assert config['weight']['fwhm'] == fwhm

        for key in wtc:
            assert config['weight'][key] == wtc[key]

        if 'fwhm_smooth' not in wtc:
            assert config['weight']['fwhm_smooth'] == DEFAULT_FWHM_SMOOTH

    with pytest.raises(ValueError):
        get_config({'weight': {'blah': 3}})


@pytest.mark.parametrize('model', ['am', 'wmom', 'gauss', 'coellip3'])
def test_psf_configs(model):

    psf_config = {'model': model}

    if model == 'am':
        psf_config['ntry'] = 3
    elif model == 'wmom':
        psf_config['weight_fwhm'] = 1.2
    else:
        psf_config['ntry'] = 3
        psf_config['lm_pars'] = {
            "maxfev": 4000, "ftol": 1.0e-5, "xtol": 1.0e-5,
        }

    config = get_config({'psf': psf_config})

    if model == 'am':
        assert 'ntry' in config['psf']
        assert config['psf']['ntry'] == psf_config['ntry']
    elif model == 'wmom':
        assert 'weight_fwhm' in config['psf']
        assert config['psf']['weight_fwhm'] == psf_config['weight_fwhm']
    else:
        assert 'ntry' in config['psf']
        assert 'lm_pars' in config['psf']

        assert config['psf']['ntry'] == psf_config['ntry']

        for key in config['psf']['lm_pars']:
            assert config['psf']['lm_pars'][key] == psf_config['lm_pars'][key]


@pytest.mark.parametrize('model', ['am', 'wmom', 'gauss', 'coellip3'])
def test_psf_configs_bad(model):

    with pytest.raises(ValueError):
        get_config({'psf': {}})

    psf_config = {'model': model}

    if model == 'am':
        # lm_pars not allowed for adaptive moments
        psf_config['lm_pars'] = {
            "maxfev": 4000, "ftol": 1.0e-5, "xtol": 1.0e-5,
        }
    elif model == 'wmom':
        # ntry not allowed for wmom
        psf_config['ntry'] = 3
    else:
        # just leave out lm_pars
        psf_config['blah'] = 3

    with pytest.raises(ValueError):
        get_config({'psf': psf_config})


def test_detect_config():
    in_config = {'detect': {'thresh': 5}}
    config = get_config()
    assert config['detect']['thresh'] == in_config['detect']['thresh']

    with pytest.raises(ValueError):
        get_config({'detect': {'blah': 5}})
