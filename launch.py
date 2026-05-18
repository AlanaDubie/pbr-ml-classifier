# ── launch.py ────────────────────────────────────────────────
# Entry point for the PBR Material Classifier tool in Maya.
#
# Run this from the Script Editor, or paste into a shelf button.
# importlib.reload() picks up any code changes without restarting Maya.
# ─────────────────────────────────────────────────────────────

import importlib
import pbr_tools
import tool_window as ui

importlib.reload(pbr_tools)
importlib.reload(ui)

# Global reference prevents Python's garbage collector from
# destroying the widget silently after this script exits.
_window = None

def show_ui():
    global _window
    _window = ui.ToolWindow()
    return _window

show_ui()