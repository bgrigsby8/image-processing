#!/usr/bin/env python3
"""
Launcher for the checkerboard distortion + lateral-CA calibration CLI.

    venv/bin/python scripts/calibrate_distortion.py --help

The implementation lives in ``src/models/calibrate_distortion.py`` so the
tests can import it the same way they import the rest of the module; this
shim only puts ``src`` on ``sys.path`` the way ``run.sh`` does.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from models.calibrate_distortion import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
