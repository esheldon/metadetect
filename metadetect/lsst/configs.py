from copy import deepcopy

from .defaults import DEFAULT_MDET_CONFIG


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
        for key in config:
            if key not in DEFAULT_MDET_CONFIG:
                raise ValueError('bad key in mdet config: %s' % key)

            # deal with sub-configs
            if key in ['detect', 'metacal', 'pgauss']:
                new_config[key].update(config[key])
            else:
                new_config[key] = config[key]

    # do some verification
    # note we allow ngmix.metacal.get_all_metacal to do its own verification
    _verify_pgauss_config(new_config['pgauss'])
    _verify_detect_config(new_config['detect'])

    return new_config


def _verify_detect_config(config):

    name = 'detect'
    _check_required_keywords(
        config=config, required_keys=['thresh'], name=name,
    )

    _check_keywords(config=config, allowed_keys=['thresh'], name=name)


def _verify_pgauss_config(config):

    name = 'pgauss'
    _check_required_keywords(
        config=config,
        required_keys=['fwhm'],
        name=name,
    )
    _check_keywords(
        config=config,
        allowed_keys=['fwhm'],
        name=name,
    )


def _check_required_keywords(config, required_keys, name):
    for key in required_keys:
        if key not in config:
            raise ValueError(f'key "{key}" must be present in {name} config')


def _check_keywords(config, allowed_keys, name):
    for key in config:
        if key not in allowed_keys:
            raise ValueError(f'key "{key}" not allowed in {name} config')
