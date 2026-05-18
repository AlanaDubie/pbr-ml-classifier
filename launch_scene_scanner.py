# ── launch_scene_scanner.py ───────────────────────────────────
# Entry point for the PBR Material Classifier tool in Maya.
#
# Run this script from Maya's Script Editor to open the tool.
# Paste the contents into a shelf button for easy access.
#
# importlib.reload() ensures that any changes made to the UI
# or scanner files are picked up without restarting Maya.
# ─────────────────────────────────────────────────────────────

import importlib
import SceneScanner
import SceneScannerUI as ui

# Reload both modules so code changes are picked up immediately
# without needing to restart Maya
importlib.reload(SceneScanner)
importlib.reload(ui)

# Store the window in a global variable.
# Without this, Python's garbage collector would silently destroy
# the widget after the function returns and the window would vanish.
_window = None

def show_ui():
    global _window
    _window = ui.SceneScannerUI()
    return _window

show_ui()