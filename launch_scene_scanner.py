import importlib
import SceneScanner as ss
import SceneScannerUI as ui

importlib.reload(ss)
importlib.reload(ui)

# Store in a global so Python doesn't garbage collect it
_window = None

def show_ui():
    global _window
    _window = ui.SceneScannerUI()
    return _window

show_ui()