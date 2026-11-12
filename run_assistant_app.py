"""Launcher for the 8ФМ desktop assistant.

Autostart points at this file when the app runs from source.  When packaged by
PyInstaller the executable itself is used instead (see :mod:`eightfm.autostart`).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eightfm.desktop_app import main

if __name__ == "__main__":
    raise SystemExit(main())
