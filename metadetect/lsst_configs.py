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

    if 'weight' not in config or config['weight'] is None:
        config['weight'] = get_default_weight_config(meas_type)


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
