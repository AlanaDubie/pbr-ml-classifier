from PySide6 import QtWidgets, QtCore
from maya import OpenMayaUI as omui
from shiboken6 import wrapInstance
import maya.cmds as cmds

from SceneScanner import SceneScanner


def get_maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


class SceneScannerUI(QtWidgets.QWidget):

    def __init__(self, parent=get_maya_main_window()):
        super().__init__(parent)

        self.scanner = SceneScanner()
        self.objects = []

        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("Scene Scanner (MVP)")
        self.setMinimumWidth(400)

        layout = QtWidgets.QVBoxLayout(self)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.scan_sel_btn = QtWidgets.QPushButton("Scan Selection")
        self.scan_sel_btn.clicked.connect(self.scan_selection)

        self.scan_scene_btn = QtWidgets.QPushButton("Scan Scene")
        self.scan_scene_btn.clicked.connect(self.scan_scene)

        btn_layout.addWidget(self.scan_sel_btn)
        btn_layout.addWidget(self.scan_scene_btn)

        layout.addLayout(btn_layout)

        # List UI
        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)

        # Status label
        self.status = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status)

    # -----------------------
    # Scan functions
    # -----------------------

    def scan_selection(self):
        self.objects = self.scanner.get_selected_meshes()
        self.populate_list()
        self.status.setText(f"Selected Scan: {len(self.objects)} objects")

        print("---- Selected Objects ----")
        for obj in self.objects:
            print(obj)

    def scan_scene(self):
        self.objects = self.scanner.get_all_scene_meshes()
        self.populate_list()
        self.status.setText(f"Scene Scan: {len(self.objects)} objects")

        print("---- Scene Objects ----")
        for obj in self.objects:
            print(obj)

    # -----------------------
    # UI helper
    # -----------------------

    def populate_list(self):
        self.list_widget.clear()

        if not self.objects:
            self.list_widget.addItem("No objects found")
            return

        for obj in self.objects:
            short = obj.split("|")[-1]
            self.list_widget.addItem(short)