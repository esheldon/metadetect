"""
test configs

note we allow the get_all_metacal to do its own verifiction
of the metacal sub config
"""
import pytest

from metadetect.lsst.configs import get_config
from metadetect.lsst.defaults import DEFAULT_STAMP_SIZE


def test_configs_smoke():
    config = get_config()

    # make sure the default is verified
    get_config(config)

    with pytest.raises(ValueError):
        get_config({'blah': 3})


@pytest.mark.parametrize('stamp_size', [None, 39])
def test_stamp_size_config(stamp_size):
    inconfig = {}

    if stamp_size is not None:
        inconfig['stamp_size'] = stamp_size

    config = get_config(inconfig)
    if stamp_size is None:
        assert config['stamp_size'] == DEFAULT_STAMP_SIZE
    else:
        assert config['stamp_size'] == stamp_size


def test_pgauss_config():
    # make sure the default is verified
    get_config()
    inconfig = {}
    get_config(inconfig)

    fwhm = 1.2
    pgauss_conf = {'fwhm': fwhm}

    inconfig = {'pgauss': pgauss_conf}
    config = get_config(inconfig)
    assert config['pgauss']['fwhm'] == fwhm

    for key in pgauss_conf:
        assert config['pgauss'][key] == pgauss_conf[key]

    with pytest.raises(ValueError):
        get_config({'pgauss': {'blah': 3}})


def test_detect_config():
    in_config = {'detect': {'thresh': 5}}
    config = get_config()
    assert config['detect']['thresh'] == in_config['detect']['thresh']

    with pytest.raises(ValueError):
        get_config({'detect': {'blah': 5}})
