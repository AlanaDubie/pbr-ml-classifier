# ── tool_window.py ────────────────────────────────────────────
# PySide6 tool window for the PBR ML Classifier.
# Parented to Maya's main window so it behaves as a native panel.
#
# Layout (top to bottom):
#   1. Scan buttons      — Scan Scene / Scan Selection
#   2. Organize button   — Organize Textures (disabled until scan runs)
#   3. Batch controls    — Accept >90%, Reject <50%, Reset All, Dry Run toggle
#   4. Filter dropdown   — filter results by material category
#   5. Results table     — Object | Material | Confidence | Status
#   6. Footer counts     — live accepted / rejected / pending totals
#   7. Detail panel      — texture path, scores, override dropdown
#   8. Status bar        — current operation or last result count
#
# Review flow:
#   Scan → predictions stored (no scene writes yet)
#   Items >= 90% confidence auto-accepted, rest stay pending
#   Artist reviews table, clicks Status to cycle, uses Override in detail panel
#   Override shown as "rock *" in Material column — asterisk = manually corrected
#   Confidence column shows "manual" when override is set
#   Hover tooltip on overridden row shows the original prediction
#   Click Organize Textures → only accepted rows get metadata + files moved
# ─────────────────────────────────────────────────────────────

import os
import time

from PySide6 import QtWidgets, QtCore
from maya import OpenMayaUI as omui
from shiboken6 import wrapInstance
import maya.cmds as cmds

from pbr_tools import PBRTools, CLASSES

# The full list of filter options shown in the dropdown.
CATEGORIES = ["all", "wood", "rock", "metal", "ground", "fabric"]

# Items at or above this confidence auto-accept after scanning.
# Items below start as pending so the artist reviews them.
AUTO_ACCEPT_THRESHOLD = 0.90

# Cycling order when the artist clicks the Status column.
STATUS_CYCLE = {"pending": "accepted", "accepted": "rejected", "rejected": "pending"}

# Column indices — defined once so changes only happen here
COL_OBJECT     = 0
COL_MATERIAL   = 1
COL_CONFIDENCE = 2
COL_STATUS     = 3


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


class ToolWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        if parent is None:
            parent = get_maya_main_window()
        super().__init__(parent)

        # The tools object handles all Maya logic — the UI just calls its methods
        self.tools = PBRTools()

        # Flat list of result dicts built after each scan, used to populate the table.
        # Each entry stores the transform key and static scan-time values.
        # The albedo_path is stored here AND read live from self.tools.results
        # at click time — live read keeps the detail panel current after files move.
        # Each entry also stores:
        #   override    — None, or a string if the artist changed the label
        #   status      — "pending" | "accepted" | "rejected"
        #   albedo_path — needed by apply_approved() to know which file to move
        self.all_results = []

        # Index of the entry currently shown in the detail panel.
        # None means the detail panel is hidden.
        self._detail_index = None

        # Whether the override combo signal is currently connected.
        # Disconnected during population to avoid spurious firings.
        self._override_connected = False

        # Which category the filter dropdown is currently set to
        self.active_filter = "all"

        self.setWindowTitle("PBR Material Classifier")
        self.setMinimumWidth(600)
        self.setMinimumHeight(800)
        self.resize(600, 800)

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

        # ── Scan buttons ──────────────────────────────────────
        scan_row = QtWidgets.QHBoxLayout()

        self.scan_scene_btn = QtWidgets.QPushButton("Scan Scene")
        self.scan_scene_btn.setToolTip(
            "Classify all mesh objects in the scene.\n"
            "Results appear in the review table — nothing is written until you organize."
        )
        self.scan_scene_btn.clicked.connect(self.run_scan_scene)

        self.scan_selection_btn = QtWidgets.QPushButton("Scan Selection")
        self.scan_selection_btn.setToolTip(
            "Classify only the currently selected mesh objects.\n"
            "Results appear in the review table — nothing is written until you organize."
        )
        self.scan_selection_btn.clicked.connect(self.run_scan_selection)

        scan_row.addWidget(self.scan_scene_btn)
        scan_row.addWidget(self.scan_selection_btn)
        root.addLayout(scan_row)

        root.addWidget(self._make_separator())

        # ── Output path + Organize ────────────────────────────
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
            "Write material tags to shaders and move textures on disk\n"
            "for Accepted items only.\n"
            "Rejected and Pending items are not touched.\n"
            "Maya's file texture paths will be updated automatically."
        )
        self.organize_btn.setEnabled(False)
        self.organize_btn.clicked.connect(self._on_organize_clicked)
        root.addWidget(self.organize_btn)

        root.addWidget(self._make_separator())

        # ── Batch controls ────────────────────────────────────
        batch_row = QtWidgets.QHBoxLayout()

        self.accept_high_btn = QtWidgets.QPushButton("Accept >90%")
        self.accept_high_btn.setToolTip(
            "Set status to Accepted for all items with confidence above 90%"
        )
        self.accept_high_btn.setEnabled(False)
        self.accept_high_btn.clicked.connect(self._batch_accept_high)

        self.reject_low_btn = QtWidgets.QPushButton("Reject <50%")
        self.reject_low_btn.setToolTip(
            "Set status to Rejected for all items with confidence below 50%"
        )
        self.reject_low_btn.setEnabled(False)
        self.reject_low_btn.clicked.connect(self._batch_reject_low)

        self.reset_btn = QtWidgets.QPushButton("Reset All")
        self.reset_btn.setToolTip("Reset all statuses to Pending and clear all overrides")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._batch_reset)

        self.dry_run_chk = QtWidgets.QCheckBox("Dry Run")
        self.dry_run_chk.setToolTip(
            "When checked, Organize Textures logs what would happen\n"
            "to the Script Editor but makes no changes to the scene or disk."
        )
        self.dry_run_chk.stateChanged.connect(self._update_footer)

        batch_row.addWidget(self.accept_high_btn)
        batch_row.addWidget(self.reject_low_btn)
        batch_row.addWidget(self.reset_btn)
        batch_row.addStretch()
        batch_row.addWidget(self.dry_run_chk)
        root.addLayout(batch_row)

        # ── Filter dropdown ───────────────────────────────────
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
        # Four columns: Object | Material | Confidence | Status
        #
        # Material column shows the predicted label normally.
        # When an override is set, shows "rock *" — asterisk = manually corrected.
        # Hovering the Material cell shows the original prediction.
        # Confidence column shows "manual" when an override is set.
        #
        # Status column cycles Pending → Accepted → Rejected on click.
        # All other columns open the detail panel on click.

        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["Object", "Material", "Confidence", "Status"])
        self.table.setColumnWidth(COL_OBJECT,     150)
        self.table.setColumnWidth(COL_MATERIAL,   180)
        self.table.setColumnWidth(COL_CONFIDENCE, 100)
        self.table.setColumnWidth(COL_STATUS,      75)
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setToolTip(
            "Click the Status column to cycle: Pending → Accepted → Rejected\n"
            "Click any other column to open the detail panel.\n"
            "Use Override in the detail panel to correct a wrong prediction.\n"
            "An asterisk (*) in Material means manually overridden.\n"
            "'manual' in Confidence means human-corrected, not ML-predicted."
        )
        self.table.itemClicked.connect(self._on_table_clicked)
        root.addWidget(self.table, stretch=1)

        # ── Footer counts ─────────────────────────────────────
        self.footer_label = QtWidgets.QLabel("—")
        self.footer_label.setAlignment(QtCore.Qt.AlignLeft)
        root.addWidget(self.footer_label)

        root.addWidget(self._make_separator())

        # ── Detail panel ──────────────────────────────────────
        # Hidden until the artist clicks a row.
        # Override dropdown lives here — not in the table — so the table
        # stays clean. When changed, the Material column updates to "rock *"
        # and Confidence updates to "manual".

        self.detail_group = QtWidgets.QGroupBox("Details")
        detail_layout = QtWidgets.QFormLayout(self.detail_group)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(8)

        self.detail_object = QtWidgets.QLabel("—")
        self.detail_shader = QtWidgets.QLabel("—")

        self.detail_path = QtWidgets.QLabel("—")
        self.detail_path.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.detail_path.setWordWrap(True)

        self.detail_scores = QtWidgets.QLabel("—")
        self.detail_scores.setWordWrap(True)

        # Override dropdown — pick a different label if the ML prediction is wrong.
        # "— keep —" means use the original prediction.
        # Changing this shows "label *" in Material and "manual" in Confidence.
        self.override_combo = QtWidgets.QComboBox()
        self.override_combo.addItem("— keep —")
        for cls in CLASSES:
            self.override_combo.addItem(cls)
        self.override_combo.setToolTip(
            "Change the predicted label for this object.\n"
            "Material column will show the corrected label with an asterisk (*).\n"
            "Confidence column will show 'manual'.\n"
            "Hover the Material cell to see the original prediction.\n"
            "The corrected label is used when organizing textures."
        )

        detail_layout.addRow("Object:",   self.detail_object)
        detail_layout.addRow("Shader:",   self.detail_shader)
        detail_layout.addRow("Texture:",  self.detail_path)
        detail_layout.addRow("Scores:",   self.detail_scores)
        detail_layout.addRow("Override:", self.override_combo)

        self.detail_group.setVisible(False)
        root.addWidget(self.detail_group)

        # ── Status bar ────────────────────────────────────────
        root.addWidget(self._make_separator())
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft)
        root.addWidget(self.status_label)

    def _make_separator(self):
        """Return a thin horizontal line used as a visual divider."""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # ── Scan actions ──────────────────────────────────────────

    def run_scan_scene(self):
        """Collect every mesh in the scene then classify them."""
        self.tools.get_all_scene_meshes()
        self._run_classify()

    def run_scan_selection(self):
        """Collect only the selected meshes then classify them."""
        self.tools.get_selected_meshes()
        self._run_classify()

    def _run_classify(self):
        """
        Run the full classification pipeline on whatever objects the scanner collected.
        Stores predictions in self.all_results — nothing written to scene yet.
        Auto-accepts high confidence results so artist only reviews uncertain ones.
        """
        total = len(self.tools.objects)

        if total == 0:
            self.status_label.setText("No mesh objects found.")
            return

        self.table.clear()
        self.all_results    = []
        self._detail_index  = None
        self.detail_group.setVisible(False)
        self.organize_btn.setEnabled(False)
        self.accept_high_btn.setEnabled(False)
        self.reject_low_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.footer_label.setText("—")
        self.status_label.setText(f"Scanning 0 / {total}...")
        QtWidgets.QApplication.processEvents()

        def on_progress(current, total, object_name):
            """Called by scan_and_classify() after each object is processed."""
            self.status_label.setText(f"Scanning {current} / {total} — {object_name}")
            QtWidgets.QApplication.processEvents()

        t0      = time.monotonic()
        results = self.tools.scan_and_classify(progress_callback=on_progress)
        elapsed = time.monotonic() - t0

        # Build a flat list of entry dicts for the table.
        # albedo_path is stored here so apply_approved() can find which
        # file to move for each accepted entry. It is also read live from
        # self.tools.results at detail-panel click time so the displayed
        # path stays current after organize has moved files.
        for transform, data in results.items():
            short      = transform.split("|")[-1]
            confidence = data.get("confidence", 0.0)
            label      = data.get("label", "unknown")

            # High confidence + valid label → auto-accept so artist only
            # needs to review uncertain items rather than approving everything
            if confidence >= AUTO_ACCEPT_THRESHOLD and label not in ("unknown", "error"):
                initial_status = "accepted"
            else:
                initial_status = "pending"

            self.all_results.append({
                "transform":   transform,
                "short":       short,
                "label":       label,
                "confidence":  confidence,
                "all_scores":  data.get("all_scores", {}),
                "shader":      data.get("shader", "—"),
                "override":    None,
                "status":      initial_status,
                "albedo_path": data.get("albedo_path"),   # required by apply_approved()
            })

        self._on_scan_complete(total, elapsed)

    def _on_scan_complete(self, total, elapsed):
        """Enable controls and populate table after scan finishes."""
        object_word = "object" if total == 1 else "objects"
        self.status_label.setText(
            f"Scan complete — {total} {object_word} in {elapsed:.1f}s"
        )

        if not self.output_path_field.text().strip():
            scene_path = cmds.file(query=True, sceneName=True) or ""
            if scene_path:
                default_dir = os.path.join(os.path.dirname(scene_path), "textures")
            else:
                project_root = cmds.workspace(query=True, rootDirectory=True) or ""
                default_dir  = os.path.join(project_root, "sourceimages", "textures")
            self.output_path_field.setText(os.path.normpath(default_dir))

        self.organize_btn.setEnabled(True)
        self.accept_high_btn.setEnabled(True)
        self.reject_low_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self._populate_table()
        self._update_footer()

    # ── Table display ─────────────────────────────────────────

    def _populate_table(self):
        """
        Fill the results table from self.all_results, applying the active filter.

        Material column:
          - Normal prediction: "wood"
          - After override:    "rock *"  (asterisk = manually corrected)
            Tooltip on the cell shows the original prediction.

        Confidence column:
          - Normal:          "99.4%"
          - After override:  "manual"  (human-corrected, not ML-predicted)

        Status column shows current status. Clicking it cycles the status.
        Each row stores its index into self.all_results via UserRole.
        """
        self.table.clear()

        for i, entry in enumerate(self.all_results):
            effective_label = entry.get("override") or entry["label"]
            if self.active_filter != "all" and effective_label != self.active_filter:
                continue

            # Confidence cell — "manual" when overridden, percentage otherwise
            if entry["override"]:
                confidence_str = "manual"
            else:
                confidence_str = (
                    f"{entry['confidence'] * 100:.1f}%"
                    if entry["confidence"] > 0 else "—"
                )

            # Material cell — asterisk signals manual correction
            if entry["override"]:
                material_str = f"{entry['override']} *"
            else:
                material_str = entry["label"]

            row = QtWidgets.QTreeWidgetItem([
                entry["short"],
                material_str,
                confidence_str,
                entry["status"].capitalize(),
            ])

            row.setData(0, QtCore.Qt.UserRole, i)

            # Tooltip on overridden Material cell shows what the original was
            if entry["override"]:
                row.setToolTip(COL_MATERIAL,
                    f"Manually overridden — original prediction: {entry['label']}")

            self.table.addTopLevelItem(row)

    def _refresh_row(self, result_index):
        """
        Refresh Material, Confidence, and Status cells for one row without
        rebuilding the entire table. Called after a status or override change.
        """
        entry = self.all_results[result_index]

        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            if item.data(0, QtCore.Qt.UserRole) == result_index:

                # Material cell
                if entry["override"]:
                    item.setText(COL_MATERIAL, f"{entry['override']} *")
                    item.setToolTip(COL_MATERIAL,
                        f"Manually overridden — original prediction: {entry['label']}")
                else:
                    item.setText(COL_MATERIAL, entry["label"])
                    item.setToolTip(COL_MATERIAL, "")

                # Confidence cell
                if entry["override"]:
                    item.setText(COL_CONFIDENCE, "manual")
                else:
                    item.setText(COL_CONFIDENCE,
                        f"{entry['confidence'] * 100:.1f}%"
                        if entry["confidence"] > 0 else "—"
                    )

                item.setText(COL_STATUS, entry["status"].capitalize())
                return

    def _on_filter_changed(self, selected_category):
        """Rebuild the table for the newly selected filter category."""
        self.active_filter  = selected_category
        self._detail_index  = None
        self.detail_group.setVisible(False)
        self._populate_table()

    def _on_table_clicked(self, item, column):
        """
        Status column — cycle the status for this row.
        Any other column — show the detail panel for this row.
        """
        result_index = item.data(0, QtCore.Qt.UserRole)
        if result_index is None:
            return

        if column == COL_STATUS:
            entry           = self.all_results[result_index]
            entry["status"] = STATUS_CYCLE[entry["status"]]
            self._refresh_row(result_index)
            self._update_footer()
        else:
            self._show_detail(result_index)

    # ── Batch controls ────────────────────────────────────────

    def _batch_accept_high(self):
        """Accept all items with confidence at or above 90%."""
        for entry in self.all_results:
            if entry["confidence"] >= AUTO_ACCEPT_THRESHOLD and \
               entry["label"] not in ("unknown", "error"):
                entry["status"] = "accepted"
        self._populate_table()
        self._update_footer()

    def _batch_reject_low(self):
        """Reject all items with confidence below 50%."""
        for entry in self.all_results:
            if 0 < entry["confidence"] < 0.50:
                entry["status"] = "rejected"
        self._populate_table()
        self._update_footer()

    def _batch_reset(self):
        """Reset all statuses to Pending and clear all overrides."""
        for entry in self.all_results:
            entry["status"]   = "pending"
            entry["override"] = None
        self._detail_index = None
        self.detail_group.setVisible(False)
        self._populate_table()
        self._update_footer()

    def _update_footer(self, *_):
        """
        Refresh footer counts and update the Organize button text to
        show how many items will actually be processed.
        """
        accepted = sum(1 for e in self.all_results if e["status"] == "accepted")
        rejected = sum(1 for e in self.all_results if e["status"] == "rejected")
        pending  = sum(1 for e in self.all_results if e["status"] == "pending")

        self.footer_label.setText(
            f"{accepted} accepted  ·  {rejected} rejected  ·  {pending} pending"
        )

        dry = self.dry_run_chk.isChecked()
        n   = accepted
        self.organize_btn.setText(
            f"Dry Run — would organize {n} item{'s' if n != 1 else ''}"
            if dry else
            f"Organize Textures ({n} accepted)"
        )

    # ── Detail panel ──────────────────────────────────────────

    def _show_detail(self, result_index):
        """
        Populate and show the detail panel for the given entry.

        Disconnects the override combo before populating to prevent
        _on_override_changed from firing during population.
        Reconnects after population is complete.
        """
        self._detail_index = result_index
        entry     = self.all_results[result_index]
        transform = entry["transform"]

        # Read live data — albedo_path may have changed if files were moved
        live_data = self.tools.results.get(transform, {})

        scores = entry.get("all_scores", {})
        score_str = (
            "\n".join(
                f"{cat}: {val * 100:.1f}%"
                for cat, val in sorted(scores.items(), key=lambda x: -(x[1] or 0))
            ) if scores else "—"
        )

        self.detail_object.setText(entry["short"])
        self.detail_shader.setText(live_data.get("shader") or entry.get("shader") or "—")
        self.detail_path.setText(live_data.get("albedo_path") or "no texture connected")
        self.detail_scores.setText(score_str)

        # Disconnect before changing selection to prevent spurious signal
        if self._override_connected:
            self.override_combo.currentTextChanged.disconnect(self._on_override_changed)
            self._override_connected = False

        if entry["override"]:
            idx = self.override_combo.findText(entry["override"])
            self.override_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self.override_combo.setCurrentIndex(0)   # "— keep —"

        self.override_combo.currentTextChanged.connect(self._on_override_changed)
        self._override_connected = True

        self.detail_group.setVisible(True)

    def _on_override_changed(self, text):
        """
        Called when the artist changes the Override dropdown.

        Stores the chosen label and immediately updates the Material column
        to show "label *" and Confidence to show "manual" so corrections
        are visible in the table without keeping the detail panel open.
        """
        if self._detail_index is None:
            return

        entry             = self.all_results[self._detail_index]
        entry["override"] = None if text == "— keep —" else text

        self._refresh_row(self._detail_index)
        self._update_footer()

    # ── Browse ────────────────────────────────────────────────

    def _on_browse_clicked(self):
        """Open a folder picker and write the chosen path into the path field."""
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

    # ── Organize Textures ─────────────────────────────────────

    def _on_organize_clicked(self):
        """
        Triggered when the artist clicks Organize Textures.

        Only processes items with status == "accepted".
        Rejected and Pending items are not touched.

        Flow:
          1. Count accepted items — bail early if none.
          2. Validate the destination folder.
          3. One confirmation dialog.
          4. Call pbr_tools.apply_approved() — writes metadata + moves files.
          5. Report result in the status bar.
        """
        dry_run  = self.dry_run_chk.isChecked()
        accepted = [e for e in self.all_results if e["status"] == "accepted"]

        if not accepted:
            self.status_label.setText(
                "No accepted items — set at least one row to Accepted first."
            )
            return

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
        n          = len(accepted)
        word       = "item" if n == 1 else "items"
        dry_note   = "\n\nDry Run is ON — no files will be moved or scene modified." if dry_run else ""

        confirm = QtWidgets.QMessageBox(self)
        confirm.setWindowTitle("Organize Textures")
        confirm.setText(f"{'[DRY RUN] ' if dry_run else ''}Organize {n} accepted {word}?")
        confirm.setInformativeText(
            f"For each accepted item this will:\n"
            f"  1. Write materialType + mlConfidence to the shader node\n"
            f"  2. Move the texture into {chosen_dir}\\<category>\\\n"
            f"  3. Update Maya's file texture paths automatically"
            f"{dry_note}\n\n"
            f"Rejected and Pending items will not be touched.\n"
            f"{'This cannot be undone.' if not dry_run else ''}"
        )

        ok_btn     = confirm.addButton(
            "Run Dry Run" if dry_run else "Organize",
            QtWidgets.QMessageBox.AcceptRole
        )
        cancel_btn = confirm.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        confirm.setDefaultButton(cancel_btn)
        confirm.exec()

        if confirm.clickedButton() == cancel_btn:
            return

        self.organize_btn.setEnabled(False)
        self.status_label.setText(
            f"{'[DRY RUN] ' if dry_run else ''}Organizing {n} {word}..."
        )
        QtWidgets.QApplication.processEvents()

        def on_progress(current, total, name):
            self.status_label.setText(
                f"{'[DRY RUN] ' if dry_run else ''}Organizing {current} / {total} — {name}"
            )
            QtWidgets.QApplication.processEvents()

        summary = self.tools.apply_approved(
            review_queue      = self.all_results,
            output_dir        = chosen_dir,
            dry_run           = dry_run,
            progress_callback = on_progress,
        )

        tagged  = summary.get("metadata_written", 0)
        moved   = summary.get("files_moved",      0)
        skipped = summary.get("skipped",          0)
        failed  = summary.get("failed",           0)

        prefix = "[DRY RUN] " if dry_run else ""
        parts  = []
        if tagged:  parts.append(f"{tagged} tagged")
        if moved:   parts.append(f"{moved} moved")
        if skipped: parts.append(f"{skipped} skipped")
        if failed:  parts.append(f"{failed} failed")

        self.status_label.setText(
            f"{prefix}Done — " + (", ".join(parts) if parts else "nothing applied")
        )

        # Refresh detail panel path if open — texture may have moved
        if self.detail_group.isVisible() and self._detail_index is not None:
            entry     = self.all_results[self._detail_index]
            live_data = self.tools.results.get(entry["transform"], {})
            self.detail_path.setText(live_data.get("albedo_path") or "no texture connected")

        self.organize_btn.setEnabled(True)
        self._update_footer()

        if failed > 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Some items failed",
                f"{failed} file{'s' if failed != 1 else ''} could not be moved.\n"
                f"Check the Script Editor for details."
            )