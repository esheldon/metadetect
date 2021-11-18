"""
test configs

note we allow the get_all_metacal to do its own verifiction
of the metacal sub config
"""
import pytest

from metadetect.lsst.configs import get_config


def test_config_smoke():
    config = get_config()

    # make sure the default is verified
    get_config(config)

    with pytest.raises(ValueError):
        get_config({'blah': 3})


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
