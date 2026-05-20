# ── tool_window.py ────────────────────────────────────────────
# PySide6 tool window for the PBR ML Classifier.
# Parented to Maya's main window so it behaves as a native panel.
#
# Layout (top to bottom):
#   1. Scan buttons       — Scan Scene / Scan Selection
#   2. Output path field  — destination folder + Browse button
#   3. Batch controls     — Accept >90%, Reject <50%, Reset, Dry Run toggle
#   4. Review table       — Object | Predicted | Confidence | Override | Status
#   5. Footer counts      — accepted / rejected / pending live totals
#   6. Approve & Organize — applies accepted items only (metadata + file move)
#   7. Detail panel       — texture path + per-class confidence scores
#   8. Status bar         — current operation or last result summary
#
# Review flow:
#   Scan → predictions stored (no scene writes yet)
#   Artist reviews table, changes overrides, sets status per row
#   Click "Approve & Organize" → only accepted rows get metadata written
#   and textures moved on disk
# ─────────────────────────────────────────────────────────────

import os
import time

from PySide6 import QtWidgets, QtCore
from maya import OpenMayaUI as omui
from shiboken6 import wrapInstance
import maya.cmds as cmds

from pbr_tools import PBRTools, CLASSES

# Confidence threshold above which a result is auto-accepted after scanning.
# Items at or above this value start with status="accepted".
# Items below start with status="pending" so the artist reviews them.
AUTO_ACCEPT_THRESHOLD = 0.90

# Cycling order when the artist clicks a status chip in the table.
STATUS_CYCLE = {"pending": "accepted", "accepted": "rejected", "rejected": "pending"}

# Filter options for the dropdown above the table.
FILTER_OPTIONS = ["all"] + CLASSES


def get_maya_main_window():
    """
    Return Maya's main application window as a Qt widget.

    Maya runs its own Qt application. Parenting our tool window to it
    means the window stays on top of Maya, minimizes with it, and
    behaves like a native panel rather than a separate floating app.

    MQtUtil.mainWindow() returns a raw C++ pointer to Maya's window.
    wrapInstance() converts it into a Python Qt object we can parent to.
    """
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


class ToolWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        if parent is None:
            parent = get_maya_main_window()
        super().__init__(parent)

        # Handles all Maya-side logic — mesh collection, classification,
        # metadata writing, and file organization.
        self.tools = PBRTools()

        # Review queue — one dict per scanned object.
        # Built after each scan, read by the table and apply_approved().
        # Each entry:
        #   transform  — full Maya node path (used as a unique key)
        #   short      — node name only (displayed in the table)
        #   label      — original ML prediction
        #   confidence — ML confidence score (0.0 – 1.0)
        #   all_scores — full per-class breakdown dict
        #   override   — None, or a string if the artist changed the label
        #   status     — "pending" | "accepted" | "rejected"
        self.review_queue = []

        # Which category the filter dropdown is set to
        self.active_filter = "all"

        self.setWindowTitle("PBR Material Classifier")
        self.setMinimumWidth(550)
        self.setMinimumHeight(800)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self._build_ui()
        self.show()

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        """
        Build and arrange all widgets in the window.
        Called once during __init__ — never rebuilt after that.
        """

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── Scan buttons ──────────────────────────────────────

        scan_row = QtWidgets.QHBoxLayout()

        self.scan_scene_btn = QtWidgets.QPushButton("Scan Scene")
        self.scan_scene_btn.setToolTip(
            "Classify all mesh objects in the scene.\n"
            "Results appear in the review table — nothing is written until you approve."
        )
        self.scan_scene_btn.clicked.connect(self.run_scan_scene)

        self.scan_selection_btn = QtWidgets.QPushButton("Scan Selection")
        self.scan_selection_btn.setToolTip(
            "Classify only the currently selected mesh objects.\n"
            "Results appear in the review table — nothing is written until you approve."
        )
        self.scan_selection_btn.clicked.connect(self.run_scan_selection)

        scan_row.addWidget(self.scan_scene_btn)
        scan_row.addWidget(self.scan_selection_btn)
        root.addLayout(scan_row)

        root.addWidget(self._make_separator())

        # ── Output path ───────────────────────────────────────
        # Where textures will be moved on disk when the artist approves.

        root.addWidget(QtWidgets.QLabel("Move textures to:"))

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

        root.addWidget(self._make_separator())

        # ── Batch controls ────────────────────────────────────
        # One-click operations that set status on multiple rows at once.
        # Dry Run toggle prevents any scene or disk changes when on.

        batch_row = QtWidgets.QHBoxLayout()

        self.accept_high_btn = QtWidgets.QPushButton("Accept >90%")
        self.accept_high_btn.setToolTip(
            f"Set status to Accepted for all items with confidence above {AUTO_ACCEPT_THRESHOLD*100:.0f}%"
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

        # Dry Run checkbox — sits at the right of the batch row
        self.dry_run_chk = QtWidgets.QCheckBox("Dry Run")
        self.dry_run_chk.setToolTip(
            "When checked, clicking Approve & Organize logs what would happen\n"
            "to the Script Editor but makes no changes to the scene or disk.\n"
            "Uncheck to apply for real."
        )

        batch_row.addWidget(self.accept_high_btn)
        batch_row.addWidget(self.reject_low_btn)
        batch_row.addWidget(self.reset_btn)
        batch_row.addStretch()
        batch_row.addWidget(self.dry_run_chk)
        root.addLayout(batch_row)

        # ── Filter dropdown ───────────────────────────────────

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Show:"))

        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(FILTER_OPTIONS)
        self.filter_combo.setToolTip("Filter the review table by material category")
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.filter_combo)
        filter_row.addStretch()
        root.addLayout(filter_row)

        # ── Review table ──────────────────────────────────────
        # Five columns:
        #   Object     — Maya node name
        #   Predicted  — ML prediction label
        #   Confidence — ML confidence score
        #   Override   — QComboBox to change the label before applying
        #   Status     — click to cycle between Pending / Accepted / Rejected
        #
        # The Override column uses setItemWidget() to embed a QComboBox
        # directly in each cell. This lets artists pick a different category
        # without leaving the table.
        #
        # The Status column shows a plain text label. Clicking a row calls
        # _on_status_clicked() which cycles the status and refreshes the row.

        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(
            ["Object", "Predicted", "Confidence", "Override", "Status"]
        )
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 90)
        self.table.setColumnWidth(3, 100)
        self.table.setColumnWidth(4, 70)
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)   # disabled — override widgets break sorting
        self.table.setToolTip(
            "Click the Status column to cycle: Pending → Accepted → Rejected\n"
            "Use the Override dropdown to change the predicted label before applying."
        )
        self.table.itemClicked.connect(self._on_table_clicked)
        root.addWidget(self.table, stretch=1)

        # ── Footer counts ─────────────────────────────────────
        # Live totals showing how many items are in each status.
        # Updated every time a status changes.

        self.footer_label = QtWidgets.QLabel("—")
        self.footer_label.setAlignment(QtCore.Qt.AlignLeft)
        root.addWidget(self.footer_label)

        # ── Approve & Organize button ─────────────────────────
        # Disabled until a scan has been run.
        # Only processes items with status == "accepted".
        # Writes metadata to shader nodes AND moves texture files on disk.

        self.approve_btn = QtWidgets.QPushButton("Approve & Organize")
        self.approve_btn.setToolTip(
            "Write material tags to shaders and move textures on disk\n"
            "for all Accepted items only.\n\n"
            "Rejected and Pending items are not touched.\n"
            "Enable Dry Run to preview without making any changes."
        )
        self.approve_btn.setEnabled(False)
        self.approve_btn.clicked.connect(self._on_approve_clicked)
        root.addWidget(self.approve_btn)

        root.addWidget(self._make_separator())

        # ── Detail panel ──────────────────────────────────────
        # Hidden until the artist clicks a row.
        # Shows the texture path and per-class confidence scores.

        self.detail_group = QtWidgets.QGroupBox("Details")
        detail_layout = QtWidgets.QFormLayout(self.detail_group)
        detail_layout.setContentsMargins(6, 6, 6, 6)
        detail_layout.setSpacing(6)

        self.detail_object = QtWidgets.QLabel("—")
        self.detail_shader = QtWidgets.QLabel("—")

        self.detail_path = QtWidgets.QLabel("—")
        self.detail_path.setWordWrap(True)
        self.detail_path.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.detail_scores = QtWidgets.QLabel("—")
        self.detail_scores.setWordWrap(True)

        detail_layout.addRow("Object:",  self.detail_object)
        detail_layout.addRow("Shader:",  self.detail_shader)
        detail_layout.addRow("Texture:", self.detail_path)
        detail_layout.addRow("Scores:",  self.detail_scores)

        self.detail_group.setVisible(False)
        root.addWidget(self.detail_group)

        # ── Status bar ────────────────────────────────────────

        root.addWidget(self._make_separator())
        self.status_label = QtWidgets.QLabel("Ready — scan a scene or selection to begin")
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
        """Collect every mesh in the scene then run classification."""
        self.tools.get_all_scene_meshes()
        self._run_classify()

    def run_scan_selection(self):
        """Collect only the selected meshes then run classification."""
        self.tools.get_selected_meshes()
        self._run_classify()

    def _run_classify(self):
        """
        Run the ML classification pipeline on self.tools.objects.

        After this call:
          - self.tools.results holds raw predictions (no scene writes)
          - self.review_queue holds one entry per object with a status
            pre-set based on confidence (>=90% → accepted, else pending)
          - The table is populated and the artist can review before applying
        """

        total = len(self.tools.objects)

        if total == 0:
            self.status_label.setText("No mesh objects found.")
            return

        # Clear previous state
        self.table.clear()
        self.review_queue   = []
        self.detail_group.setVisible(False)
        self.approve_btn.setEnabled(False)
        self.accept_high_btn.setEnabled(False)
        self.reject_low_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.footer_label.setText("—")
        self.status_label.setText(f"Scanning 0 / {total}...")
        QtWidgets.QApplication.processEvents()

        def on_progress(current, total, object_name):
            self.status_label.setText(f"Scanning {current} / {total} — {object_name}")
            QtWidgets.QApplication.processEvents()

        t0      = time.monotonic()
        results = self.tools.scan_and_classify(progress_callback=on_progress)
        elapsed = time.monotonic() - t0

        # Build the review queue from the raw results.
        # Auto-accept high confidence results so the artist only needs to
        # review the uncertain ones — this is the "smart default" pattern
        # used in production pipeline tools.
        for transform, data in results.items():
            confidence = data.get("confidence", 0.0)
            label      = data.get("label", "unknown")

            # High confidence and valid label → auto-accept
            # Low confidence or unknown/error → leave as pending for review
            if confidence >= AUTO_ACCEPT_THRESHOLD and label not in ("unknown", "error"):
                initial_status = "accepted"
            else:
                initial_status = "pending"

            self.review_queue.append({
                "transform":   transform,
                "short":       transform.split("|")[-1],
                "label":       label,
                "confidence":  confidence,
                "all_scores":  data.get("all_scores", {}),
                "override":    None,
                "status":      initial_status,
                "albedo_path": data.get("albedo_path"),
            })

        self._on_scan_complete(total, elapsed)

    def _on_scan_complete(self, total, elapsed):
        """
        Called after _run_classify() finishes.
        Enables controls, sets a default output path, and populates the table.
        """

        object_word = "object" if total == 1 else "objects"
        self.status_label.setText(
            f"Scan complete — {total} {object_word} in {elapsed:.1f}s  "
        )

        # Suggest a default output path only if the field is empty
        if not self.output_path_field.text().strip():
            scene_path = cmds.file(query=True, sceneName=True) or ""
            if scene_path:
                default_dir = os.path.join(os.path.dirname(scene_path), "textures")
                self.output_path_field.setText(os.path.normpath(default_dir))

        self.approve_btn.setEnabled(True)
        self.accept_high_btn.setEnabled(True)
        self.reject_low_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        self._populate_table()
        self._update_footer()

    # ── Table display ─────────────────────────────────────────

    def _populate_table(self):
        """
        Fill the review table from self.review_queue, applying the active filter.

        For each row:
          - Columns 0-2 (Object, Predicted, Confidence) are plain text
          - Column 3 (Override) gets a QComboBox embedded via setItemWidget()
          - Column 4 (Status) is plain text — clicking cycles the status

        The index into self.review_queue is stored on each row via UserRole
        so _on_table_clicked() and _on_override_changed() can find the entry.
        """

        # Temporarily disconnect itemClicked to prevent it firing while
        # we rebuild rows (setItemWidget triggers internal signals).
        self.table.itemClicked.disconnect(self._on_table_clicked)
        self.table.clear()

        if self.active_filter == "all":
            visible = self.review_queue
        else:
            visible = [e for e in self.review_queue if (e.get("override") or e["label"]) == self.active_filter]

        for queue_index, entry in enumerate(self.review_queue):
            # Apply filter — skip rows that don't match
            effective_label = entry.get("override") or entry["label"]
            if self.active_filter != "all" and effective_label != self.active_filter:
                continue

            confidence_str = (
                f"{entry['confidence'] * 100:.1f}%"
                if entry["confidence"] > 0 else "—"
            )

            row = QtWidgets.QTreeWidgetItem([
                entry["short"],
                entry["label"],
                confidence_str,
                "",            # Override column — widget set below
                entry["status"].capitalize(),
            ])

            # Store the queue index so event handlers can find this entry
            row.setData(0, QtCore.Qt.UserRole, queue_index)

            self.table.addTopLevelItem(row)

            # Embed a QComboBox in the Override column.
            # "— keep —" means use the original ML prediction.
            # Any other choice replaces the label when applying.
            override_combo = QtWidgets.QComboBox()
            override_combo.addItem("— keep —")
            for cls in CLASSES:
                override_combo.addItem(cls)

            # Restore the artist's previous override selection if there is one
            if entry["override"]:
                idx = override_combo.findText(entry["override"])
                if idx >= 0:
                    override_combo.setCurrentIndex(idx)

            # Pass the queue index into the lambda so it survives the loop
            override_combo.currentTextChanged.connect(
                lambda text, qi=queue_index: self._on_override_changed(qi, text)
            )

            self.table.setItemWidget(row, 3, override_combo)

        self.table.itemClicked.connect(self._on_table_clicked)

    def _on_filter_changed(self, selected):
        """Rebuild the table for the newly selected filter category."""
        self.active_filter = selected
        self.detail_group.setVisible(False)
        self._populate_table()

    def _on_table_clicked(self, item, column):
        """
        Handles all clicks on the review table.

        Column 4 (Status) — cycle the status for this row
        Any other column  — show the detail panel for this row
        """

        queue_index = item.data(0, QtCore.Qt.UserRole)
        if queue_index is None:
            return

        if column == 4:
            # Cycle the status and refresh just this row's status cell
            entry          = self.review_queue[queue_index]
            entry["status"] = STATUS_CYCLE[entry["status"]]
            item.setText(4, entry["status"].capitalize())
            self._update_footer()
        else:
            # Show the detail panel for this object
            self._show_detail(queue_index)

    def _on_override_changed(self, queue_index, text):
        """
        Called when the artist changes the Override dropdown for a row.
        Stores None (keep original) or the chosen category string.
        Also refreshes the footer in case the filter changes visibility.
        """
        entry             = self.review_queue[queue_index]
        entry["override"] = None if text == "— keep —" else text
        self._update_footer()

    # ── Batch controls ────────────────────────────────────────

    def _batch_accept_high(self):
        """Accept all items with confidence >= 90%."""
        for entry in self.review_queue:
            if entry["confidence"] >= AUTO_ACCEPT_THRESHOLD and \
               entry["label"] not in ("unknown", "error"):
                entry["status"] = "accepted"
        self._populate_table()
        self._update_footer()

    def _batch_reject_low(self):
        """Reject all items with confidence below 50%."""
        for entry in self.review_queue:
            if entry["confidence"] < 0.50:
                entry["status"] = "rejected"
        self._populate_table()
        self._update_footer()

    def _batch_reset(self):
        """Reset all statuses to pending and clear all overrides."""
        for entry in self.review_queue:
            entry["status"]   = "pending"
            entry["override"] = None
        self._populate_table()
        self._update_footer()

    def _update_footer(self):
        """
        Update the footer label with live accepted/rejected/pending counts.
        Also updates the Approve & Organize button text to show how many
        items will actually be processed.
        """

        accepted = sum(1 for e in self.review_queue if e["status"] == "accepted")
        rejected = sum(1 for e in self.review_queue if e["status"] == "rejected")
        pending  = sum(1 for e in self.review_queue if e["status"] == "pending")

        self.footer_label.setText(
            f"{accepted} accepted  ·  {rejected} rejected  ·  {pending} pending"
        )

        dry = self.dry_run_chk.isChecked()
        label = (
            f"Dry Run — would apply {accepted} item{'s' if accepted != 1 else ''}"
            if dry else
            f"Approve & Organize ({accepted} item{'s' if accepted != 1 else ''})"
        )
        self.approve_btn.setText(label)

    # ── Detail panel ──────────────────────────────────────────

    def _show_detail(self, queue_index):
        """
        Populate the detail panel for the given queue entry.
        Reads albedo_path live from self.tools.results to always show
        the current path even after files have been moved.
        """

        entry     = self.review_queue[queue_index]
        transform = entry["transform"]

        # Read the live result dict — albedo_path may have been updated
        # by apply_approved() if this row was already processed.
        live_data = self.tools.results.get(transform, {})

        scores = entry.get("all_scores", {})
        if scores:
            score_lines = [
                f"{cat}: {val * 100:.1f}%"
                for cat, val in sorted(scores.items(), key=lambda x: -(x[1] or 0))
            ]
            score_str = "\n".join(score_lines)
        else:
            score_str = "—"

        self.detail_object.setText(entry["short"])
        self.detail_shader.setText(live_data.get("shader") or "—")
        self.detail_path.setText(live_data.get("albedo_path") or "no texture connected")
        self.detail_scores.setText(score_str)
        self.detail_group.setVisible(True)

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

    # ── Approve & Organize ────────────────────────────────────

    def _on_approve_clicked(self):
        """
        Triggered when the artist clicks 'Approve & Organize'.

        Flow:
          1. Count accepted items — bail early if none.
          2. Validate the destination folder.
          3. One confirmation dialog.
          4. Call pbr_tools.apply_approved() which handles both metadata
             writing and file organization in one pass.
          5. Report the result in the status bar.

        Dry Run mode skips all scene/disk changes and logs to Script Editor.
        """

        dry_run  = self.dry_run_chk.isChecked()
        accepted = [e for e in self.review_queue if e["status"] == "accepted"]

        # ── Step 1: check there is something to apply ─────────

        if not accepted:
            self.status_label.setText(
                "No accepted items — set at least one row to Accepted first."
            )
            return

        # ── Step 2: validate the destination folder ───────────

        chosen_dir = self.output_path_field.text().strip()

        if not chosen_dir:
            self.status_label.setText(
                "Enter a destination folder before organizing."
            )
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

        # ── Step 3: confirmation dialog ───────────────────────

        item_word = "item" if len(accepted) == 1 else "items"
        dry_note  = "\n\nDry Run is ON — no files will be moved or scene modified." if dry_run else ""

        confirm = QtWidgets.QMessageBox(self)
        confirm.setWindowTitle("Approve & Organize")
        confirm.setText(
            f"{'[DRY RUN] ' if dry_run else ''}Apply {len(accepted)} approved {item_word}?"
        )
        confirm.setInformativeText(
            f"For each accepted item this will:\n"
            f"  1. Write materialType + mlConfidence to the shader node\n"
            f"  2. Move the texture file into {chosen_dir}\\<category>\\\n"
            f"  3. Update Maya's file texture paths automatically"
            f"{dry_note}\n\n"
            f"Rejected and Pending items will not be touched.\n"
            f"{'This cannot be undone.' if not dry_run else ''}"
        )

        ok_label  = "Run Dry Run" if dry_run else "Apply"
        ok_btn    = confirm.addButton(ok_label, QtWidgets.QMessageBox.AcceptRole)
        cancel_btn = confirm.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        confirm.setDefaultButton(cancel_btn)
        confirm.exec()

        if confirm.clickedButton() == cancel_btn:
            return

        # ── Step 4: run apply_approved() ─────────────────────

        self.approve_btn.setEnabled(False)
        self.status_label.setText(
            f"{'[DRY Run] ' if dry_run else ''}Applying {len(accepted)} {item_word}..."
        )
        QtWidgets.QApplication.processEvents()

        def on_progress(current, total, name):
            self.status_label.setText(
                f"{'[DRY RUN] ' if dry_run else ''}Applying {current} / {total} — {name}"
            )
            QtWidgets.QApplication.processEvents()

        summary = self.tools.apply_approved(
            review_queue      = self.review_queue,
            output_dir        = chosen_dir,
            dry_run           = dry_run,
            progress_callback = on_progress,
        )

        tagged  = summary.get("metadata_written", 0)
        moved   = summary.get("files_moved",      0)
        skipped = summary.get("skipped",          0)
        failed  = summary.get("failed",           0)

        # ── Step 5: report result in status bar ───────────────

        prefix = "[DRY RUN] " if dry_run else ""
        parts  = []
        if tagged:
            parts.append(f"{tagged} tagged")
        if moved:
            parts.append(f"{moved} moved")
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")

        self.status_label.setText(
            f"{prefix}Done — " + (", ".join(parts) if parts else "nothing applied")
        )

        # Refresh the detail panel path if it's open — the texture may
        # have moved, and self.tools.results was updated by apply_approved().
        if self.detail_group.isVisible():
            shown = self.detail_object.text()
            for entry in self.review_queue:
                if entry["short"] == shown:
                    live_data = self.tools.results.get(entry["transform"], {})
                    self.detail_path.setText(
                        live_data.get("albedo_path") or "no texture connected"
                    )
                    break

        self.approve_btn.setEnabled(True)
        self._update_footer()

        # Only interrupt the artist with a dialog if something went wrong
        if failed > 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Some items failed",
                f"{failed} file{'s' if failed != 1 else ''} could not be moved.\n"
                f"Check the Script Editor for details."
            )