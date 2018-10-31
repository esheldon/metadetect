import os
from glob import glob
from distutils.core import setup

setup(
    name="metadetect", 
    version="0.1.0",
    description="Combining detection and metacalibration",
    license = "GPL",
    author="Erin Scott Sheldon",
    author_email="erin.sheldon@gmail.com",
    packages=['metadetect'],
)
