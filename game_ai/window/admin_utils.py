from __future__ import annotations

import ctypes
import os


def is_running_as_admin() -> bool:
    if os.name != "nt":
        return True

    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except OSError:
        return False
