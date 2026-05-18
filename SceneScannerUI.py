# ── SceneScannerUI.py ────────────────────────────────────────
# PySide6 tool window for the PBR ML Classifier.
# Parented to Maya's main window so it behaves as a native panel.
#
# Layout (top to bottom):
#   1. Scan buttons      — Scan Scene / Scan Selection
#   2. Organize button   — Organize Textures (disabled until scan runs)
#   3. Filter dropdown   — filter results by material category
#   4. Results table     — object name, predicted label, confidence
#   5. Detail panel      — texture path + per-class score breakdown
#   6. Status bar        — current operation or last result count
# ─────────────────────────────────────────────────────────────

import os
import time

from PySide6 import QtWidgets, QtCore
from maya import OpenMayaUI as omui
from shiboken6 import wrapInstance
import maya.cmds as cmds

from SceneScanner import SceneScanner

# The full list of filter options shown in the dropdown.
# "all" shows every result; the rest filter by that material category.
CATEGORIES = ["all", "wood", "rock", "metal", "ground", "fabric"]


def get_maya_main_window():
    """
    Return Maya's main application window as a Qt widget.

    Why do we need this?
    Maya runs its own Qt application internally. If we create our tool
    window without parenting it to Maya's window, it behaves like a
    completely separate application — it won't stay on top, it won't
    minimize with Maya, and it can get lost behind other windows.

    MQtUtil.mainWindow() returns a raw C++ pointer to Maya's window.
    wrapInstance() converts that pointer into a Python Qt object we
    can use as a parent.
    """
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


class SceneScannerUI(QtWidgets.QWidget):

    def __init__(self, parent=None):
        if parent is None:
            parent = get_maya_main_window()
        super().__init__(parent)

        # The scanner handles all Maya logic — the UI just calls its methods
        self.scanner = SceneScanner()

        # Flat list of result dicts built after each scan, used to populate the table
        self.all_results = []

        # Which category the filter dropdown is currently set to
        self.active_filter = "all"

        self.setWindowTitle("PBR Material Classifier")
        self.setMinimumWidth(420)
        self.setMinimumHeight(650)

        # Qt.Window makes this a proper floating window instead of an embedded widget
        self.setWindowFlags(QtCore.Qt.Window)

        # WA_DeleteOnClose frees memory when the window is closed
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self._build_ui()
        self.show()

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        """
        Build and arrange all widgets in the window.
        Called once during __init__ — widgets are never rebuilt after that.
        """

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Scan buttons ─────────────────────────────────────
        # Two buttons side by side — scan the whole scene or just the selection

        scan_row = QtWidgets.QHBoxLayout()

        self.scan_scene_btn = QtWidgets.QPushButton("Scan Scene")
        self.scan_scene_btn.setToolTip("Classify all mesh objects in the scene")
        self.scan_scene_btn.clicked.connect(self.run_scan_scene)

        self.scan_selection_btn = QtWidgets.QPushButton("Scan Selection")
        self.scan_selection_btn.setToolTip("Classify only the currently selected mesh objects")
        self.scan_selection_btn.clicked.connect(self.run_scan_selection)

        scan_row.addWidget(self.scan_scene_btn)
        scan_row.addWidget(self.scan_selection_btn)
        root.addLayout(scan_row)

        root.addWidget(self._make_separator())

        # ── Output path + Organize ────────────────────────────
        # Text field so the artist can see and edit the destination folder.
        # The Browse button opens a folder picker as a convenience.
        # Organize Textures is disabled until a scan has been run.

        path_label = QtWidgets.QLabel("Move textures to:")
        root.addWidget(path_label)

        path_row = QtWidgets.QHBoxLayout()

        self.output_path_field = QtWidgets.QLineEdit()
        self.output_path_field.setPlaceholderText("Choose or type a destination folder...")
        self.output_path_field.setToolTip(
            "Textures will be moved into category subfolders inside this folder.\n"
            "e.g. <folder>/wood/   <folder>/metal/   etc."
        )
        path_row.addWidget(self.output_path_field)

        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.setFixedWidth(60)
        self.browse_btn.setToolTip("Open a folder picker to choose the destination")
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        path_row.addWidget(self.browse_btn)

        root.addLayout(path_row)

        self.organize_btn = QtWidgets.QPushButton("Organize Textures")
        self.organize_btn.setToolTip(
            "Move classified textures into category subfolders inside the output folder.\n"
            "Maya's file texture paths will be updated automatically."
        )
        self.organize_btn.setEnabled(False)   # enabled in _on_scan_complete()
        self.organize_btn.clicked.connect(self._on_organize_clicked)
        root.addWidget(self.organize_btn)

        root.addWidget(self._make_separator())

        # ── Filter dropdown ───────────────────────────────────
        # Lets the artist narrow the table to one material category

        filter_row = QtWidgets.QHBoxLayout()
        filter_label = QtWidgets.QLabel("Show:")
        filter_row.addWidget(filter_label)

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(CATEGORIES)
        self.filter_combo.setToolTip("Filter results by material category")
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        root.addLayout(filter_row)

        # ── Results table ─────────────────────────────────────
        # Shows one row per scanned object with its predicted label and confidence

        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["Object", "Material", "Confidence"])
        self.table.setColumnWidth(0, 180)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 80)
        self.table.setRootIsDecorated(False)      # flat list, no tree arrows
        self.table.setAlternatingRowColors(True)  # easier to read with alternating colours
        self.table.setSortingEnabled(True)        # click column headers to sort
        self.table.setToolTip("Click a row to see texture path and per-class scores")
        self.table.itemClicked.connect(self._on_row_clicked)
        root.addWidget(self.table, stretch=1)     # stretch=1 lets the table expand to fill space

        # ── Detail panel ──────────────────────────────────────
        # Hidden until the artist clicks a row in the table.
        # Shows the full texture path and every category's confidence score.

        self.detail_group = QtWidgets.QGroupBox("Details")
        detail_layout = QtWidgets.QFormLayout(self.detail_group)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(8)

        self.detail_object = QtWidgets.QLabel("—")
        self.detail_shader = QtWidgets.QLabel("—")

        # Word wrap allows long file paths to break across lines
        self.detail_path = QtWidgets.QLabel("—")
        self.detail_path.setWordWrap(True)

        self.detail_scores = QtWidgets.QLabel("—")
        self.detail_scores.setWordWrap(True)

        detail_layout.addRow("Object:",  self.detail_object)
        detail_layout.addRow("Shader:",  self.detail_shader)
        detail_layout.addRow("Texture:", self.detail_path)
        detail_layout.addRow("Scores:",  self.detail_scores)

        self.detail_group.setVisible(False)
        root.addWidget(self.detail_group)

        # ── Status bar ────────────────────────────────────────
        # One line of text at the bottom showing what the tool is doing

        root.addWidget(self._make_separator())
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft)
        root.addWidget(self.status_label)

    def _make_separator(self):
        """
        Return a thin horizontal line widget used as a visual divider
        between sections of the UI.
        """
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # ── Scan actions ──────────────────────────────────────────

    def run_scan_scene(self):
        """
        Triggered when the artist clicks 'Scan Scene'.
        Tells the scanner to collect every mesh in the scene, then classifies them.
        """
        self.scanner.get_all_scene_meshes()
        self._run_classify()

    def run_scan_selection(self):
        """
        Triggered when the artist clicks 'Scan Selection'.
        Tells the scanner to collect only the selected meshes, then classifies them.
        """
        self.scanner.get_selected_meshes()
        self._run_classify()

    def _run_classify(self):
        """
        Run the full classification pipeline on whatever objects the scanner collected.
        Updates the status bar as each object is processed, then populates the table.
        """

        total = len(self.scanner.objects)

        if total == 0:
            self.status_label.setText("No mesh objects found.")
            return

        # Clear the previous results before the new scan starts
        self.table.clear()
        self.all_results = []
        self.detail_group.setVisible(False)
        self.organize_btn.setEnabled(False)   # disable until the new scan completes
        self.status_label.setText(f"Scanning 0 / {total}...")

        # processEvents() forces the UI to redraw right now so the artist
        # sees the status update instead of a frozen window during the scan
        QtWidgets.QApplication.processEvents()

        def on_progress(current, total, object_name):
            """
            Called by scan_and_classify() after each object is processed.
            Updates the status bar so the artist can see progress in real time.
            """
            self.status_label.setText(f"Scanning {current} / {total} — {object_name}")
            QtWidgets.QApplication.processEvents()

        t0      = time.monotonic()
        results = self.scanner.scan_and_classify(progress_callback=on_progress)
        elapsed = time.monotonic() - t0

        # Convert the results dict into a flat list of entry dicts for the table.
        # We store everything the UI might need so we don't have to re-query
        # the scanner when the artist clicks a row or changes the filter.
        for transform, data in results.items():
            short = transform.split("|")[-1]   # just the node name, not the full path
            entry = {
                "transform":   transform,
                "short":       short,
                "label":       data.get("label", "unknown"),
                "confidence":  data.get("confidence", 0.0),
                "albedo_path": data.get("albedo_path", ""),
                "all_scores":  data.get("all_scores", {}),
                "shader":      data.get("shader", "—"),
            }
            self.all_results.append(entry)

        self._on_scan_complete(total, elapsed)

    def _on_scan_complete(self, total, elapsed):
        """
        Called at the end of _run_classify() once all results are ready.
        Updates the status bar, populates a default output path, enables
        the Organize button, and fills the table.
        """

        object_word = "object" if total == 1 else "objects"
        self.status_label.setText(
            f"Scan complete — {total} {object_word} in {elapsed:.1f}s"
        )

        # Populate a sensible default output path if the field is still empty.
        # Don't overwrite it if the artist already typed something.
        if not self.output_path_field.text().strip():
            scene_path = cmds.file(query=True, sceneName=True) or ""
            if scene_path:
                default_dir = os.path.join(os.path.dirname(scene_path), "textures")
            else:
                project_root = cmds.workspace(query=True, rootDirectory=True) or ""
                default_dir  = os.path.join(project_root, "sourceimages", "textures")
            self.output_path_field.setText(os.path.normpath(default_dir))

        # Now that we have results, the artist can organize the textures
        self.organize_btn.setEnabled(True)

        self._populate_table()

    # ── Table display ─────────────────────────────────────────

    def _populate_table(self):
        """
        Fill the results table from self.all_results, applying the active filter.
        Called after every scan and every time the filter dropdown changes.
        """

        self.table.clear()

        # Apply the active category filter.
        # "all" shows everything; any other value shows only that category.
        if self.active_filter == "all":
            visible_results = self.all_results
        else:
            visible_results = [
                r for r in self.all_results if r["label"] == self.active_filter
            ]

        for entry in visible_results:
            # Format the confidence as a percentage string, or "—" if zero
            if entry["confidence"] > 0:
                confidence_str = f"{entry['confidence'] * 100:.1f}%"
            else:
                confidence_str = "—"

            row = QtWidgets.QTreeWidgetItem([
                entry["short"],
                entry["label"],
                confidence_str,
            ])

            # Store the full entry dict on the row so _on_row_clicked()
            # can retrieve it without searching through all_results again
            row.setData(0, QtCore.Qt.UserRole, entry)

            self.table.addTopLevelItem(row)

    def _on_filter_changed(self, selected_category):
        """
        Triggered when the artist changes the filter dropdown.
        Rebuilds the table to show only the selected category.
        """
        self.active_filter = selected_category
        self.detail_group.setVisible(False)   # hide detail panel when filter changes
        self._populate_table()

    # ── Detail panel ──────────────────────────────────────────

    def _on_row_clicked(self, item):
        """
        Triggered when the artist clicks a row in the results table.
        Populates the detail panel with the texture path and per-class scores.
        """

        entry = item.data(0, QtCore.Qt.UserRole)
        if not entry:
            return

        self.detail_object.setText(entry["short"])
        self.detail_shader.setText(entry.get("shader") or "—")
        self.detail_path.setText(entry.get("albedo_path") or "no texture connected")

        # Build the scores string — sorted highest to lowest so the
        # most likely categories appear at the top
        scores = entry.get("all_scores", {})
        if scores:
            score_lines = [
                f"{category}: {value * 100:.1f}%"
                for category, value in sorted(scores.items(), key=lambda x: -(x[1] or 0))
            ]
            score_str = "\n".join(score_lines)
        else:
            score_str = "—"

        self.detail_scores.setText(score_str)
        self.detail_group.setVisible(True)

    # ── Organize Textures ─────────────────────────────────────

    def _on_browse_clicked(self):
        """
        Opens a folder picker and writes the chosen path into the output path field.
        Uses whatever is already in the field as the starting directory.
        """
        current = self.output_path_field.text().strip()
        start   = current if os.path.isdir(current) else ""

        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose textures output folder",
            start,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )

        if chosen:
            self.output_path_field.setText(os.path.normpath(chosen))

    def _on_organize_clicked(self):
        """
        Triggered when the artist clicks 'Organize Textures'.

        Flow:
          1. Check there are classified textures to move.
          2. Read the destination path from the path field.
          3. One confirmation dialog — shows the path, Move Files / Cancel.
          4. Run and report result in the status bar.
             Only shows a second dialog if files actually failed to move.
        """

        # ── Step 1: check there is something to move ─────────

        movable_paths = set()
        for data in self.scanner.results.values():
            path  = data.get("albedo_path")
            label = data.get("label")
            if path and label not in (None, "unknown", "error"):
                movable_paths.add(path)

        texture_count = len(movable_paths)

        if texture_count == 0:
            self.status_label.setText("Nothing to organize — run a scan first.")
            return

        # ── Step 2: read and validate the destination path ───

        chosen_dir = self.output_path_field.text().strip()

        if not chosen_dir:
            self.status_label.setText("Enter a destination folder before organizing.")
            self.output_path_field.setFocus()
            return

        if not os.path.isdir(chosen_dir):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Folder does not exist",
                f"This folder doesn't exist yet:\n{chosen_dir}\n\nCreate it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Cancel,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            os.makedirs(chosen_dir, exist_ok=True)

        chosen_dir = os.path.normpath(chosen_dir)

        # ── Step 3: one confirmation dialog ──────────────────

        file_word = "file" if texture_count == 1 else "files"

        confirm = QtWidgets.QMessageBox(self)
        confirm.setWindowTitle("Organize Textures")
        confirm.setText(f"Move {texture_count} texture {file_word}?")
        confirm.setInformativeText(
            f"To:  {chosen_dir}\\<category>\\\n\n"
            f"Maya's texture paths will be updated automatically.\n"
            f"This cannot be undone."
        )

        move_button   = confirm.addButton("Move Files", QtWidgets.QMessageBox.AcceptRole)
        cancel_button = confirm.addButton("Cancel",     QtWidgets.QMessageBox.RejectRole)
        confirm.setDefaultButton(cancel_button)
        confirm.exec()

        if confirm.clickedButton() == cancel_button:
            return

        # ── Step 4: run and report in the status bar ─────────

        self.organize_btn.setEnabled(False)
        self.status_label.setText("Moving textures...")
        QtWidgets.QApplication.processEvents()

        def on_organize_progress(current, total, filename):
            self.status_label.setText(f"Moving {current} / {total} — {filename}")
            QtWidgets.QApplication.processEvents()

        summary = self.scanner.organize_textures(
            output_dir=chosen_dir,
            progress_callback=on_organize_progress,
        )

        moved   = summary.get("moved",   0)
        skipped = summary.get("skipped", 0)
        failed  = summary.get("failed",  0)

        # Refresh detail panel paths from updated results
        for entry in self.all_results:
            updated  = self.scanner.results.get(entry["transform"], {})
            new_path = updated.get("albedo_path")
            if new_path:
                entry["albedo_path"] = new_path

        # Report in the status bar — no popup needed on success
        parts = []
        if moved:
            parts.append(f"{moved} moved")
        if skipped:
            parts.append(f"{skipped} already organized")
        if failed:
            parts.append(f"{failed} failed")
        self.status_label.setText("Done — " + (", ".join(parts) if parts else "nothing to move"))

        self.organize_btn.setEnabled(True)

        # Only interrupt the artist if something actually went wrong
        if failed > 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Some files could not be moved",
                f"{failed} file{'s' if failed != 1 else ''} could not be moved.\n"
                f"Check the Script Editor for details."
            )