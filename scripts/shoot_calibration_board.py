#!/usr/bin/env python3
"""
Launcher for the arm-driven checkerboard capture (the shooting half of the
distortion calibration; ``calibrate_distortion.py`` is the fitting half).

    venv/bin/python scripts/shoot_calibration_board.py --help

The implementation lives in ``src/models/shoot_calibration_board.py`` so the
tests can import it the same way they import the rest of the module; this
shim only puts ``src`` on ``sys.path`` the way ``run.sh`` does.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from models.shoot_calibration_board import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
