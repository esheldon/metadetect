build() {
    # Work around SIP on MacOSX
   export DYLD_LIBRARY_PATH=$LSST_LIBRARY_PATH
   python setup.py pytest --addopts "metadetect/tests/ metadetect/lsst/tests/test_import.py"
   default_build
}

