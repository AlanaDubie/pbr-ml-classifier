# ── SceneScannerUI.py ────────────────────────────────────────
# PySide6 tool window for the PBR ML Classifier.
# Parented to Maya's main window so it behaves as a native panel.
# Maya-style UI
# ─────────────────────────────────────────────────────────────

from PySide6 import QtWidgets, QtCore
from maya import OpenMayaUI as omui
from shiboken6 import wrapInstance

from SceneScanner import SceneScanner

CATEGORIES = ["all", "wood", "rock", "metal", "ground", "fabric"]

def get_maya_main_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)

class SceneScannerUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        if parent is None:
            parent = get_maya_main_window()
        super().__init__(parent)

        self.scanner = SceneScanner()
        self.all_results = []
        self.active_filter = "all"

        self.setWindowTitle("PBR Material Classifier")
        self.setMinimumWidth(420)
        self.setMinimumHeight(650)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self._build_ui()
        self.show()

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # Scan buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.scan_scene_btn = QtWidgets.QPushButton("Scan Scene")
        self.scan_sel_btn   = QtWidgets.QPushButton("Scan Selection")
        self.scan_scene_btn.clicked.connect(self.run_scan_scene)
        self.scan_sel_btn.clicked.connect(self.run_scan_selection)
        btn_row.addWidget(self.scan_scene_btn)
        btn_row.addWidget(self.scan_sel_btn)
        root.addLayout(btn_row)

        # Separator
        root.addWidget(self._separator())

        # Filter row
        filter_row = QtWidgets.QHBoxLayout()
        filter_lbl = QtWidgets.QLabel("Show:")
        filter_row.addWidget(filter_lbl)
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(CATEGORIES)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        root.addLayout(filter_row)

        # Results table
        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["Object", "Material", "Confidence"])
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 80)
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.itemClicked.connect(self._on_row_clicked)
        root.addWidget(self.table, stretch=1)

        # Detail group
        self.detail_group = QtWidgets.QGroupBox("Details")
        detail_layout = QtWidgets.QFormLayout(self.detail_group)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(8)

        self.detail_object  = QtWidgets.QLabel("—")
        self.detail_shader  = QtWidgets.QLabel("—")
        self.detail_path    = QtWidgets.QLabel("—")
        self.detail_path.setWordWrap(True)
        self.detail_scores  = QtWidgets.QLabel("—")
        self.detail_scores.setWordWrap(True)

        detail_layout.addRow("Object:",   self.detail_object)
        detail_layout.addRow("Shader:",   self.detail_shader)
        detail_layout.addRow("Texture:",  self.detail_path)
        detail_layout.addRow("Scores:",   self.detail_scores)

        self.detail_group.setVisible(False)
        root.addWidget(self.detail_group)

        # Status bar
        root.addWidget(self._separator())
        self.status_lbl = QtWidgets.QLabel("Ready")
        self.status_lbl.setAlignment(QtCore.Qt.AlignLeft)
        root.addWidget(self.status_lbl)

    def _separator(self):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # ── Scan actions ──────────────────────────────────────────

    def run_scan_scene(self):
        self.scanner.get_all_scene_meshes()
        self._run_classify()

    def run_scan_selection(self):
        self.scanner.get_selected_meshes()
        self._run_classify()

    def _run_classify(self):
        total = len(self.scanner.objects)
        if total == 0:
            self.status_lbl.setText("No objects found.")
            return

        self.table.clear()
        self.all_results = []
        self.detail_group.setVisible(False)
        self.status_lbl.setText(f"Scanning 0 / {total}...")
        QtWidgets.QApplication.processEvents()

        def on_progress(current, total, name):
            self.status_lbl.setText(f"Scanning {current} / {total} — {name}")
            QtWidgets.QApplication.processEvents()

        import time
        t0      = time.monotonic()
        results = self.scanner.scan_and_classify(progress_callback=on_progress)
        elapsed = time.monotonic() - t0

        for transform, data in results.items():
            short = transform.split("|")[-1]
            conf  = data.get("confidence", 0.0)
            entry = {
                "transform":  transform,
                "short":      short,
                "label":      data.get("label", "unknown"),
                "confidence": conf,
                "albedo_path":data.get("albedo_path", ""),
                "all_scores": data.get("all_scores", {}),
                "shader":     data.get("shader", "—"),
            }
            self.all_results.append(entry)

        self.status_lbl.setText(
            f"Scan complete — {total} object{'s' if total != 1 else ''} in {elapsed:.1f}s"
        )
        self._populate_table()

    # ── Table display ─────────────────────────────────────────

    def _populate_table(self):
        self.table.clear()
        cat = self.active_filter
        visible = self.all_results if cat == "all" else [
            r for r in self.all_results if r["label"] == cat
        ]
        for entry in visible:
            conf_str = f"{entry['confidence']*100:.1f}%" if entry["confidence"] > 0 else "—"
            item = QtWidgets.QTreeWidgetItem([
                entry["short"],
                entry["label"],
                conf_str,
            ])
            item.setData(0, QtCore.Qt.UserRole, entry)
            self.table.addTopLevelItem(item)

    def _on_filter_changed(self, text):
        self.active_filter = text
        self.detail_group.setVisible(False)
        self._populate_table()

    # ── Detail panel ──────────────────────────────────────────

    def _on_row_clicked(self, item):
        entry = item.data(0, QtCore.Qt.UserRole)
        if not entry:
            return
        print("[DEBUG] all_scores:", entry.get("all_scores"))

        self.detail_object.setText(entry["short"])
        self.detail_shader.setText(entry.get("shader", "—"))
        self.detail_path.setText(entry.get("albedo_path") or "no texture connected")

        scores = entry.get("all_scores", {})
        if scores:
            score_str = "\n".join(
                f"{c}: {v*100:.1f}%"
                for c, v in sorted(scores.items(), key=lambda x: -(x[1] or 0))
            )
        else:
            score_str = "—"
        self.detail_scores.setText(score_str)

        self.detail_group.setVisible(True)