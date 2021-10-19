from copy import deepcopy

from .defaults import (
    DEFAULT_MDET_CONFIG,
    DEFAULT_WEIGHT_FWHMS,
    DEFAULT_STAMP_SIZES,
)


def get_config(config=None):
    """
    extract a full config, overriding defaults with the input

    Parameters
    ----------
    config: dict, optional
        Entries in this dict override defaults stored in
        defaults.DEFAULT_MDET_CONFIG

    Returns
    -------
    Full config
    """

    new_config = deepcopy(DEFAULT_MDET_CONFIG)

    if config is not None:
        new_config.update(deepcopy(config))

    _fill_config(new_config)

    return new_config


def _fill_config(config):
    """
    check the config and fill in some defaults programatically
    """
    # moments are not a model, use more generic description measurement_type
    # or meas_type for short.  But support old name 'model'

    if 'model' in config:
        config['meas_type'] = config.pop('model')

    for key in config:
        if key not in DEFAULT_MDET_CONFIG:
            raise ValueError('bad key in mdet config: %s' % key)

    meas_type = config['meas_type']

    if 'stamp_size' not in config or config['stamp_size'] is None:
        config['stamp_size'] = get_default_stamp_size(meas_type)

    if meas_type != 'am':
        if 'weight' not in config or config['weight'] is None:
            config['weight'] = get_default_weight_config(meas_type)

    # note we allow ngmix.metacal.get_all_metacal to do its
    # own verification
    _verify_psf_config(config['psf'])
    _verify_detect_config(config['detect'])


def get_default_weight_config(meas_type):
    """
    get the default weight function configuration based
    on the measurement type
    """
    return {'fwhm': get_default_weight_fwhm(meas_type)}


def get_default_stamp_size(meas_type):
    """
    get default stamp size for the input measurement type
    """
    if meas_type not in DEFAULT_STAMP_SIZES:
        raise ValueError('bad meas type: %s' % meas_type)

    return DEFAULT_STAMP_SIZES[meas_type]


def get_default_weight_fwhm(meas_type):
    """
    get default weight fwhm for the input measurement type
    """
    if meas_type not in DEFAULT_WEIGHT_FWHMS:
        raise ValueError('bad meas type: %s' % meas_type)

    return DEFAULT_WEIGHT_FWHMS[meas_type]


def _verify_psf_config(config):

    name = 'psf'
    _check_required_keywords(
        config=config, required_keys=['model'], name=name,
    )

    model = config['model']
    if model in ['am', 'admom']:
        allowed_keys = ['model', 'ntry']
    elif model == 'wmom':
        allowed_keys = ['model', 'weight_fwhm']
    else:
        allowed_keys = ['model', 'lm_pars', 'ntry']

    _check_keywords(
        config=config,
        allowed_keys=allowed_keys,
        name=name,
    )


def _verify_detect_config(config):

    name = 'detect'
    _check_required_keywords(
        config=config, required_keys=['thresh'], name=name,
    )

    _check_keywords(config=config, allowed_keys=['thresh'], name=name)


def _check_required_keywords(config, required_keys, name):
    for key in required_keys:
        if key not in config:
            raise ValueError(f'key "{key}" must be present in {name} config')


def _check_keywords(config, allowed_keys, name):
    for key in config:
        if key not in allowed_keys:
            raise ValueError(f'key "{key}" not allowed in {name} config')
