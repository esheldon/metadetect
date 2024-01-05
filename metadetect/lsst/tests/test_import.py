# This file is part of metadetect.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.

import unittest

from lsst.utils.tests import ImportTestCase


class MetadetectImportTestCase(ImportTestCase):
    """Test that every file can be imported.

    metadetect package has dependencies on packages that are not
    in rubin-env. Routines that needs those packages (for upstream repo)
    import them on the fly. This test case acts as a place to document those
    files that can or cannot be imported.
    """

    PACKAGES = {
        "metadetect.lsst",
        "metadetect",
    }
    SKIP_FILES = {
        "metadetect": {
            # This depends on the meds package, and are not needed for LSST.
            "detect.py",
        }
    }


if __name__ == "__main__":
    unittest.main()
