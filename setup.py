import os
from setuptools import setup, find_packages

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "metadetect",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name="metadetect",
    version=__version__,
    packages=find_packages(),
    description="Combining detection and metacalibration",
    license="GPL",
    author="Erin Scott Sheldon",
    author_email="erin.sheldon@gmail.com",
    url='https://github.com/esheldon/metadetect',
)
