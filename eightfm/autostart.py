"""Cross-platform autostart registration for the desktop assistant.

The desktop app calls :func:`set_enabled` when the user toggles autostart.  On
Windows this writes an ``HKCU\\...\\Run`` value, on Linux a freedesktop
``.desktop`` file, and on macOS a per-user LaunchAgent plist.  No admin rights
are required on any platform.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "8FM Assistant"

_WIN_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_WIN_VALUE = "8FM Assistant"
_LINUX_FILE = "8fm-assistant.desktop"
_MAC_LABEL = "com.8fm.assistant"


def _quote(part: str) -> str:
    if not part or any(ch in part for ch in ' \t"'):
        return '"' + part.replace('"', '\\"') + '"'
    return part


def launch_command(*, minimized: bool = True) -> str:
    """Build the command line that starts the desktop app.

    When running from source it points at ``run_assistant_app.py`` next to the
    package and prefers ``pythonw.exe`` so no console window appears.  When
    frozen by PyInstaller it simply re-launches the executable.
    """
    extra = ["--minimized"] if minimized else []
    if getattr(sys, "frozen", False):
        parts = [sys.executable, *extra]
    else:
        root = Path(__file__).resolve().parent.parent
        launcher = root / "run_assistant_app.py"
        python = Path(sys.executable)
        if os.name == "nt":
            windowless = python.with_name("pythonw.exe")
            if windowless.exists():
                python = windowless
        parts = [str(python), str(launcher), *extra]
    return " ".join(_quote(part) for part in parts)


def _linux_desktop_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "autostart" / _LINUX_FILE


def _mac_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_MAC_LABEL}.plist"


def is_enabled() -> bool:
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _WIN_RUN_KEY) as key:
                value, _ = winreg.QueryValueEx(key, _WIN_VALUE)
        except OSError:
            return False
        return bool(value)
    if sys.platform == "darwin":
        return _mac_plist_path().exists()
    return _linux_desktop_path().exists()


def set_enabled(enabled: bool) -> None:
    """Enable or disable autostart for the current user."""
    if os.name == "nt":
        _set_windows(enabled)
    elif sys.platform == "darwin":
        _set_mac(enabled)
    else:
        _set_linux(enabled)


def _set_windows(enabled: bool) -> None:
    import winreg

    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _WIN_RUN_KEY) as key:
        if enabled:
            winreg.SetValueEx(key, _WIN_VALUE, 0, winreg.REG_SZ, launch_command())
        else:
            try:
                winreg.DeleteValue(key, _WIN_VALUE)
            except FileNotFoundError:
                pass


def _set_linux(enabled: bool) -> None:
    path = _linux_desktop_path()
    if not enabled:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        f"Exec={launch_command()}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n",
        encoding="utf-8",
    )


def _set_mac(enabled: bool) -> None:
    path = _mac_plist_path()
    if not enabled:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import plistlib
    import shlex

    payload = {
        "Label": _MAC_LABEL,
        "ProgramArguments": shlex.split(launch_command()),
        "RunAtLoad": True,
        "KeepAlive": False,
    }
    path.write_bytes(plistlib.dumps(payload))
