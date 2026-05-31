# ── tool_window.py ────────────────────────────────────────────
# PySide6 tool window for the PBR ML Classifier.
# Parented to Maya's main window so it behaves as a native panel.
#
# Layout (top to bottom):
#   1. Scan buttons      — Scan Scene / Scan Selection
#   2. Output path       — destination folder + Browse button
#   3. Batch controls    — Show filter, Accept High, Reject Low, Reset All, Dry Run
#   4. Results table     — Object | Material | Confidence | Status
#   5. Footer counts     — live accepted / rejected / pending totals
#   6. Detail panel      — two-panel split:
#        Left  — Asset, Source folder, Shader, Classification, Override
#        Right — Maps (CollapsibleContainer) + Scores (CollapsibleContainer)
#   7. Organize button   — applies accepted items only
#   8. Status bar        — current operation or last result count
#
# Review flow:
#   Scan → predictions stored (no scene writes yet)
#   Items >= 90% confidence auto-accepted, rest stay pending
#   Artist reviews table, clicks Status to cycle, uses Override in detail panel
#   Override shown as "rock *" in Material column — asterisk = manually corrected
#   Confidence column shows High / Medium / Low / manual quality tiers
#   Hover confidence cell to see raw percentage
#   Click Organize Textures → only accepted rows get metadata + files moved
# ─────────────────────────────────────────────────────────────

import os
import time

from PySide6 import QtWidgets, QtCore, QtGui
from maya import OpenMayaUI as omui
from shiboken6 import wrapInstance
import maya.cmds as cmds

from widgets.collapsable import CollapsibleContainer
from pbr_tools import PBRTools, CLASSES

try:
    from texture_name_parser import resolve_asset_name
except ImportError:
    def resolve_asset_name(path):
        return os.path.splitext(os.path.basename(path))[0] if path else "—"


# Filter options shown in the dropdown above the table
CATEGORIES = ["all", "wood", "rock", "metal", "ground", "fabric"]

# Items at or above this confidence auto-accept after scanning.
AUTO_ACCEPT_THRESHOLD = 0.90

# Cycling order when the artist clicks the Status column.
STATUS_CYCLE = {"pending": "accepted", "accepted": "rejected", "rejected": "pending"}

# Column indices
COL_OBJECT     = 0
COL_MATERIAL   = 1
COL_CONFIDENCE = 2
COL_STATUS     = 3

CONF_HIGH   = 0.80
CONF_MEDIUM = 0.50


def confidence_tier(value):
    """Return a quality tier label for a confidence value (0.0–1.0)."""
    if value >= CONF_HIGH:     return "High"
    elif value >= CONF_MEDIUM: return "Medium"
    elif value > 0:            return "Low"
    return "—"


def apply_confidence_color(item, text):
    """Color the confidence column text based on tier."""
    colors = {
        "High":   QtGui.QColor(90, 200, 120),
        "Medium": QtGui.QColor(220, 170, 70),
        "Low":    QtGui.QColor(210, 90, 90),
        "Manual": QtGui.QColor(120, 170, 255),
    }
    color = colors.get(text)
    if color:
        item.setForeground(COL_CONFIDENCE, QtGui.QBrush(color))


def get_maya_main_window():
    """Return Maya's main window as a Qt widget for parenting."""
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


class ToolWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        if parent is None:
            parent = get_maya_main_window()
        super().__init__(parent)

        self.tools = PBRTools()

        # Flat list of result dicts built after each scan.
        # Each entry:
        #   transform, short, label, confidence, all_scores, shader,
        #   override, status, albedo_path, all_paths
        self.all_results = []

        self._detail_index       = None
        self._override_connected = False
        self.active_filter       = "all"

        self.setWindowTitle("PBR Material Classifier")
        self.setMinimumWidth(750)
        self.setMinimumHeight(950)
        self.resize(750, 950)

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
        scan_box = QtWidgets.QGroupBox("Scan")
        scan_layout = QtWidgets.QHBoxLayout(scan_box)

        self.scan_scene_btn = QtWidgets.QPushButton("Scan Scene")
        self.scan_scene_btn.setToolTip(
            "Classify all mesh objects in the scene.\n"
            "Nothing is written until you click Organize Textures."
        )
        self.scan_scene_btn.clicked.connect(self.run_scan_scene)
        scan_layout.addWidget(self.scan_scene_btn)

        self.scan_selection_btn = QtWidgets.QPushButton("Scan Selection")
        self.scan_selection_btn.setToolTip(
            "Classify only the currently selected mesh objects.\n"
            "Nothing is written until you click Organize Textures."
        )
        self.scan_selection_btn.clicked.connect(self.run_scan_selection)
        scan_layout.addWidget(self.scan_selection_btn)

        root.addWidget(scan_box)

        # ── Output Texture Folder ─────────────────────────────
        texture_folder_box = QtWidgets.QGroupBox("Output Texture Folder")
        texture_folder_layout = QtWidgets.QHBoxLayout(texture_folder_box)

        self.output_path_field = QtWidgets.QLineEdit()
        self.output_path_field.setPlaceholderText(
            "Choose or type a destination folder for your textures..."
        )
        self.output_path_field.setToolTip(
            "Textures will be moved into category subfolders inside this folder.\n"
            "e.g. <folder>/wood/   <folder>/metal/   etc."
        )
        texture_folder_layout.addWidget(self.output_path_field)

        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.setFixedWidth(60)
        self.browse_btn.setToolTip("Open a folder picker to choose the destination")
        self.browse_btn.clicked.connect(self._on_browse_clicked)
        texture_folder_layout.addWidget(self.browse_btn)

        root.addWidget(texture_folder_box)

        root.addWidget(self._make_separator())

        # ── Batch controls row ────────────────────────────────
        # Show filter | Accept High | Reject Low | Reset All | Dry Run
        batch_row = QtWidgets.QHBoxLayout()

        batch_row.addWidget(QtWidgets.QLabel("Show:"))
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(CATEGORIES)
        self.filter_combo.setToolTip("Filter results by material category")
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        batch_row.addWidget(self.filter_combo)

        batch_row.addSpacing(8)

        self.accept_high_btn = QtWidgets.QPushButton("Accept High")
        self.accept_high_btn.setToolTip(
            f"Accept all items with confidence {int(AUTO_ACCEPT_THRESHOLD * 100)}%+"
        )
        self.accept_high_btn.setEnabled(False)
        self.accept_high_btn.clicked.connect(self._batch_accept_high)
        batch_row.addWidget(self.accept_high_btn)

        self.reject_low_btn = QtWidgets.QPushButton("Reject Low")
        self.reject_low_btn.setToolTip("Reject all items with confidence below 50%")
        self.reject_low_btn.setEnabled(False)
        self.reject_low_btn.clicked.connect(self._batch_reject_low)
        batch_row.addWidget(self.reject_low_btn)

        self.reset_btn = QtWidgets.QPushButton("Reset All")
        self.reset_btn.setToolTip("Reset all statuses to Pending and clear all overrides")
        self.reset_btn.setEnabled(False)
        self.reset_btn.clicked.connect(self._batch_reset)
        batch_row.addWidget(self.reset_btn)

        batch_row.addStretch()

        self.dry_run_chk = QtWidgets.QCheckBox("Dry Run")
        self.dry_run_chk.setToolTip(
            "When checked, Organize Textures logs what would happen\n"
            "but makes no changes to the scene or disk."
        )
        self.dry_run_chk.stateChanged.connect(self._update_footer)
        batch_row.addWidget(self.dry_run_chk)

        root.addLayout(batch_row)

        # ── Results table ─────────────────────────────────────
        self.table = QtWidgets.QTreeWidget()
        self.table.setHeaderLabels(["Object", "Material", "Confidence", "Status"])
        self.table.setColumnWidth(COL_OBJECT,     190)
        self.table.setColumnWidth(COL_MATERIAL,   100)
        self.table.setColumnWidth(COL_CONFIDENCE,  90)
        self.table.setColumnWidth(COL_STATUS,      80)
        self.table.setRootIsDecorated(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setToolTip(
            "Click Status to cycle: Pending → Accepted → Rejected\n"
            "Click any other column to open the detail panel.\n"
            "Use Override in the detail panel to correct a wrong prediction.\n"
            "An asterisk (*) in Material means manually overridden.\n"
            "Hover Confidence to see the raw score."
        )
        self.table.headerItem().setToolTip(
            COL_CONFIDENCE,
            "High = 80%+   Medium = 50–80%   Low = below 50%\n"
            "Hover any row to see the exact confidence score."
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
        # Split into two columns using QSplitter:
        #   Left  — Asset / Source / Shader / Classification / Override
        #   Right — Maps (CollapsibleContainer) + Scores (CollapsibleContainer)
        #
        # CollapsibleContainer from widgets/collapsable.py uses Maya's own
        # :teDownArrow/:teRightArrow icons so headers match the Attribute Editor.

        self.detail_group = QtWidgets.QGroupBox("Details")
        detail_group_layout = QtWidgets.QVBoxLayout(self.detail_group)
        detail_group_layout.setContentsMargins(4, 6, 4, 6)
        detail_group_layout.setSpacing(0)

        # Fixed two-column 
        _detail_body = QtWidgets.QWidget()
        _detail_body_layout = QtWidgets.QHBoxLayout(_detail_body)
        _detail_body_layout.setContentsMargins(3, 3, 3, 3)
        _detail_body_layout.setSpacing(0)

        detail_group_layout.addWidget(_detail_body)

        # ── Left panel ────────────────────────────────────────
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 4, 8, 4)

        _detail_body_layout.addWidget(left_widget, stretch=2)

        left_form = QtWidgets.QFormLayout()
        left_form.setSpacing(3)
        left_form.setLabelAlignment(QtCore.Qt.AlignLeft)
        left_form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        self.detail_asset          = QtWidgets.QLabel("—")
        self.detail_source         = QtWidgets.QLabel("—")
        self.detail_shader         = QtWidgets.QLabel("—")
        self.detail_classification = QtWidgets.QLabel("—")

        left_form.addRow("Asset:", self.detail_asset)
        left_form.addRow("Path:", self.detail_source)
        left_form.addRow("Shader:", self.detail_shader)
        left_form.addRow(QtWidgets.QLabel("Classification:"))
        left_form.addRow(self.detail_classification)

        # Confidence Scores 
        # Starts collapsed so it doesn't compete with the classification line.
        # Artist expands it when they want the full per-class breakdown.
        self._scores_container = CollapsibleContainer(
            "Confidence Scores", collapsed=True, color_background=True, max_width=200
        )
        scores_inner_layout = QtWidgets.QVBoxLayout(self._scores_container.content_widget)
        scores_inner_layout.setContentsMargins(8, 4, 8, 6)

        self.detail_scores = QtWidgets.QLabel("—")
        self.detail_scores.setWordWrap(True)
        scores_inner_layout.addWidget(self.detail_scores)

        left_form.addRow(self._scores_container)
        left_layout.addLayout(left_form)

        # Override dropdown
        override_form = QtWidgets.QFormLayout()
        override_form.setSpacing(5)
        override_form.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.override_combo = QtWidgets.QComboBox()
        self.override_combo.addItem("— keep —")
        for cls in CLASSES:
            self.override_combo.addItem(cls)
        self.override_combo.setToolTip(
            "Change the predicted label for this object.\n"
            "Material column will show the corrected label with an asterisk (*).\n"
            "Confidence column will show 'Manual'.\n"
            "Hover the Material cell to see the original prediction.\n"
            "The corrected label is used when organizing textures."
        )
        override_form.addRow("Override:", self.override_combo)
        left_layout.addLayout(override_form)
        left_layout.addStretch()

        # Thin vertical divider between the two columns
        _vline = QtWidgets.QFrame()
        _vline.setFrameShape(QtWidgets.QFrame.VLine)
        _detail_body_layout.addWidget(_vline)

        # ── Right panel — Maps (plain list, always visible) ───────────────────
        # Not collapsible — maps are reference info the artist reads directly.
        # One filename per line inside a plain QLabel with word wrap.
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 4, 8, 4)

        right_layout.setSpacing(6)
        _detail_body_layout.addWidget(right_widget, stretch=3)

        _maps_header = QtWidgets.QLabel("Maps: ")
        _maps_header_font = _maps_header.font()
        _maps_header.setFont(_maps_header_font)
        right_layout.addWidget(_maps_header)

        self.detail_maps = QtWidgets.QLabel("—")
        self.detail_maps.setWordWrap(True)
        self.detail_maps.setTextFormat(QtCore.Qt.PlainText)
        self.detail_maps.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        right_layout.addWidget(self.detail_maps)
        right_layout.addStretch()

        detail_group_layout.addWidget(_detail_body)

        self.detail_group.setVisible(False)
        root.addWidget(self.detail_group)

        # ── Organize button ───────────────────────────────────
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

        # ── Status bar ────────────────────────────────────────
        self.status_label = QtWidgets.QLabel("Ready — Last scan: —")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft)
        root.addWidget(self.status_label)

    def _make_separator(self):
        """Return a thin horizontal line used as a visual divider."""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
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
        Run the full classification pipeline.
        Stores predictions in self.all_results — nothing written to scene yet.
        Auto-accepts high confidence results so artist only reviews uncertain ones.
        """
        total = len(self.tools.objects)
        if total == 0:
            self.status_label.setText("No mesh objects found.")
            return

        self.table.clear()
        self.all_results         = []
        self._detail_index       = None
        self.detail_group.setVisible(False)
        self.organize_btn.setEnabled(False)
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

        for transform, data in results.items():
            short      = transform.split("|")[-1]
            confidence = data.get("confidence", 0.0)
            label      = data.get("label", "unknown")

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
                "shader":      data.get("shader"),
                "override":    None,
                "status":      initial_status,
                "albedo_path": data.get("albedo_path"),
                "all_paths":   data.get("all_paths", []),
            })

        self._on_scan_complete(total, elapsed)

    def _on_scan_complete(self, total, elapsed):
        """Enable controls and populate table after scan finishes."""
        word = "object" if total == 1 else "objects"
        self.status_label.setText(
            f"Ready — Last scan: {total} {word} in {elapsed:.1f}s"
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

        Material  — "rock *" when override set; tooltip shows original prediction
        Confidence — High/Medium/Low tier; "Manual" when overridden; tooltip = raw %
        Status    — cycles on click; all other columns open detail panel
        """
        self.table.clear()

        for i, entry in enumerate(self.all_results):
            effective_label = entry.get("override") or entry["label"]
            if self.active_filter != "all" and effective_label != self.active_filter:
                continue

            if entry["override"]:
                conf_str     = "Manual"
                conf_tooltip = f"Manually overridden — original confidence: {entry['confidence'] * 100:.1f}%"
            else:
                conf_str     = confidence_tier(entry["confidence"])
                conf_tooltip = (
                    f"{entry['confidence'] * 100:.1f}%"
                    if entry["confidence"] > 0 else "—"
                )

            material_str = f"{entry['override']} *" if entry["override"] else entry["label"]

            row = QtWidgets.QTreeWidgetItem([
                entry["short"], material_str, conf_str, entry["status"].capitalize()
            ])
            apply_confidence_color(row, conf_str)
            row.setData(0, QtCore.Qt.UserRole, i)
            row.setToolTip(COL_CONFIDENCE, conf_tooltip)
            if entry["override"]:
                row.setToolTip(COL_MATERIAL,
                    f"Manually overridden — original prediction: {entry['label']}")

            self.table.addTopLevelItem(row)

    def _refresh_row(self, result_index):
        """Refresh Material, Confidence, and Status cells for one row."""
        entry = self.all_results[result_index]

        for i in range(self.table.topLevelItemCount()):
            item = self.table.topLevelItem(i)
            if item.data(0, QtCore.Qt.UserRole) == result_index:
                if entry["override"]:
                    item.setText(COL_MATERIAL, f"{entry['override']} *")
                    item.setToolTip(COL_MATERIAL,
                        f"Manually overridden — original prediction: {entry['label']}")
                    item.setText(COL_CONFIDENCE, "Manual")
                    item.setToolTip(COL_CONFIDENCE,
                        f"Manually overridden — original confidence: {entry['confidence'] * 100:.1f}%")
                else:
                    item.setText(COL_MATERIAL, entry["label"])
                    item.setToolTip(COL_MATERIAL, "")
                    item.setText(COL_CONFIDENCE, confidence_tier(entry["confidence"]))
                    item.setToolTip(COL_CONFIDENCE,
                        f"{entry['confidence'] * 100:.1f}%" if entry["confidence"] > 0 else "—")

                apply_confidence_color(item, item.text(COL_CONFIDENCE))
                item.setText(COL_STATUS, entry["status"].capitalize())
                return

    def _on_filter_changed(self, selected_category):
        """Rebuild the table for the newly selected filter category."""
        self.active_filter = selected_category
        self._detail_index = None
        self.detail_group.setVisible(False)
        self._populate_table()

    def _on_table_clicked(self, item, column):
        """Status column — cycle status. Any other column — open detail panel."""
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
        Refresh footer counts and Organize button text.
        Accepts *_ so it can connect directly to QCheckBox.stateChanged.
        """
        accepted = sum(1 for e in self.all_results if e["status"] == "accepted")
        rejected = sum(1 for e in self.all_results if e["status"] == "rejected")
        pending  = sum(1 for e in self.all_results if e["status"] == "pending")

        self.footer_label.setText(
            f"{accepted} accepted  •  {rejected} rejected  •  {pending} pending"
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

        Left panel  — asset, source folder, shader, classification,
                      confidence scores (CollapsibleContainer), override
        Right panel — Maps filenames (CollapsibleContainer)

        Disconnects the override combo before populating to prevent
        _on_override_changed from firing during population.
        """
        self._detail_index = result_index
        entry     = self.all_results[result_index]
        transform = entry["transform"]

        live_data   = self.tools.results.get(transform, {})
        albedo_path = live_data.get("albedo_path") or entry.get("albedo_path") or ""
        all_paths   = live_data.get("all_paths")   or entry.get("all_paths")   or []
        print(f"[DEBUG] all_paths for {entry['short']}: {all_paths}")

        # ── Left: asset name ──────────────────────────────────
        asset_name = resolve_asset_name(albedo_path) if albedo_path else "—"
        self.detail_asset.setText(asset_name)

        # ── Left: source folder ───────────────────────────────
        # Show the last two path components so long paths stay readable.
        if albedo_path:
            src_folder = os.path.dirname(albedo_path)
            parts      = src_folder.replace("\\", "/").split("/")
            short_src  = "/".join(parts[-2:]) if len(parts) >= 2 else src_folder
            self.detail_source.setText(short_src)
            self.detail_source.setToolTip(src_folder)
        else:
            self.detail_source.setText("no texture connected")
            self.detail_source.setToolTip("")

        # ── Left: shader ──────────────────────────────────────
        shader = live_data.get("shader") or entry.get("shader") or "—"
        self.detail_shader.setText(shader)

        # ── Left: classification ──────────────────────────────
        # Normal:   "rock   (0.94 confidence)"
        # Override: "fabric * (Manual)"
        label      = entry.get("override") or entry["label"]
        confidence = entry["confidence"]
        if entry["override"]:
            self.detail_classification.setText(f"{entry['override']} * (Manual)")
        elif confidence > 0:
            self.detail_classification.setText(f"{label} \n({confidence * 100:.1f}% confidence)")
        else:
            self.detail_classification.setText(label)

        # ── Left: confidence scores ───────────────────────────
        # Hide confidence breakdown when manually overridden.
        if entry.get("override"):
            self.detail_scores.setText("—")
        else:
            scores = entry.get("all_scores", {})
            if scores:
                self.detail_scores.setText(
                    "\n".join(
                        f"{cat}: {val * 100:.1f}%"
                        for cat, val in sorted(
                            scores.items(),
                            key=lambda x: -(x[1] or 0)
                        )
                    )
                )
            else:
                self.detail_scores.setText("—")

        # ── Right: maps detected ──────────────────────────────
        # One filename per line — plain text, no checkmarks or bullets.
        # Falls back to the albedo filename if no full set was found.
        if all_paths:
            self.detail_maps.setText(
                "\n".join(os.path.basename(p) for p in all_paths)
            )
        elif albedo_path:
            self.detail_maps.setText(os.path.basename(albedo_path))
        else:
            self.detail_maps.setText("—")

        # ── Override combo ────────────────────────────────────
        # Disconnect before changing selection to prevent spurious signal
        if self._override_connected:
            self.override_combo.currentTextChanged.disconnect(self._on_override_changed)
            self._override_connected = False

        if entry["override"]:
            idx = self.override_combo.findText(entry["override"])
            self.override_combo.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            self.override_combo.setCurrentIndex(0)

        self.override_combo.currentTextChanged.connect(self._on_override_changed)
        self._override_connected = True

        self.detail_group.setVisible(True)

    def _on_override_changed(self, text):
        """
        Store the chosen label and update the table row + classification line.
        Format when overridden: "fabric * (Manual)"
        """
        if self._detail_index is None:
            return

        entry = self.all_results[self._detail_index]
        entry["override"] = None if text == "— keep —" else text

        label = entry.get("override") or entry["label"]
        confidence = entry["confidence"]


        if entry["override"]:
            self.detail_classification.setText(f"{entry['override']} * (Manual)")
        elif confidence > 0:
            self.detail_classification.setText(f"{label} \n({confidence * 100:.1f}% confidence)")
        else:
            self.detail_classification.setText(label)

        self._refresh_row(self._detail_index)
        self._show_detail(self._detail_index)
        self._update_footer()

    # ── Browse ────────────────────────────────────────────────

    def _on_browse_clicked(self):
        """Open a folder picker and write the chosen path into the path field."""
        current = self.output_path_field.text().strip()
        start   = current if os.path.isdir(current) else ""

        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose textures output folder", start,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if chosen:
            self.output_path_field.setText(os.path.normpath(chosen))

    # ── Organize Textures ─────────────────────────────────────

    def _on_organize_clicked(self):
        """
        Triggered when the artist clicks Organize Textures.
        Only accepted items are processed — rejected and pending untouched.

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
                self, "Folder does not exist",
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
        dry_note   = (
            "\n\nDry Run is ON — no files will be moved or scene modified."
            if dry_run else ""
        )

        confirm = QtWidgets.QMessageBox(self)
        confirm.setWindowTitle("Organize Textures")
        confirm.setText(f"{'[DRY RUN] ' if dry_run else ''}Organize {n} accepted {word}?")
        confirm.setInformativeText(
            f"For each accepted item this will:\n"
            f"  1. Write materialType + mlConfidence to the shader node\n"
            f"  2. Move the texture set into {chosen_dir}\\<category>\\<asset>\\\n"
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

        # Refresh detail panel — source folder and maps may have changed
        if self.detail_group.isVisible() and self._detail_index is not None:
            self._show_detail(self._detail_index)

        self.organize_btn.setEnabled(True)
        self._update_footer()

        if failed > 0:
            QtWidgets.QMessageBox.warning(
                self, "Some items failed",
                f"{failed} file{'s' if failed != 1 else ''} could not be moved.\n"
                f"Check the Script Editor for details."
            )