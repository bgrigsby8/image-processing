#!/usr/bin/env python3
"""
Launcher for the focus sweep: step the camera's focus across a range, shoot
the checkerboard at each step and rank the frames by slanted-edge MTF50.

    venv/bin/python scripts/focus_sweep.py --help

The implementation lives in ``src/models/focus_sweep.py`` so the
tests can import it the same way they import the rest of the module; this
shim only puts ``src`` on ``sys.path`` the way ``run.sh`` does.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from models.focus_sweep import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
