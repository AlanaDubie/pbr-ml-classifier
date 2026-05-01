import importlib
import SceneScannerUI as ui

importlib.reload(ui)

def show_ui():
    return ui.SceneScannerUI()

show_ui()