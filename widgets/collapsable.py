# widgets/collapsable.py  
# This is a reusable widget for any Maya tool that needs collapsible sections.
#
# Provides CollapsibleContainer(name, collapsed=True, color_background=False)
#
# Credit to AndySedano for the original implementation of the collapsible section,
# repo can be found here: https://github.com/aronamao/PySide2-Collapsible-Widget/blob/main/Container.py

from PySide6 import QtWidgets, QtCore, QtGui

class _CollapsibleHeader(QtWidgets.QWidget):
    """
    Clickable header bar for a CollapsibleContainer.
 
    Uses a QStackedLayout to layer a coloured background behind the
    header row — the same technique Maya uses for its own section
    headers in the Attribute Editor and Channel Box.
 
    The expand/collapse icons are Maya's own built-in resource icons
    (:teDownArrow.png and :teRightArrow.png) so the look matches
    Maya's native panels exactly.
    """
 
    def __init__(self, name, content_widget):
        super().__init__()
        self._content = content_widget
 
        # Maya's built-in arrow icons — available in any Maya session
        self._expand_icon   = QtGui.QPixmap(":teDownArrow.png")
        self._collapse_icon = QtGui.QPixmap(":teRightArrow.png")
 
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed
        )
 
        # QStackedLayout.StackAll layers widgets on top of each other.
        # The background label sits behind the header row so we get a
        # solid coloured bar without fighting Qt's stylesheet cascade.
        stacked = QtWidgets.QStackedLayout(self)
        stacked.setStackingMode(QtWidgets.QStackedLayout.StackAll)
 
        # Bottom layer — coloured background matching Maya's section headers
        background = QtWidgets.QLabel()
        background.setStyleSheet(
            "QLabel { background-color: rgb(93, 93, 93); border-radius: 2px; }"
        )
 
        # Top layer — arrow icon + bold section name
        header_widget = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header_widget)
        header_layout.setContentsMargins(11, 0, 11, 0)
 
        self._icon = QtWidgets.QLabel()
        self._icon.setPixmap(self._expand_icon)
        header_layout.addWidget(self._icon)
 
        label_font = QtGui.QFont()
        label_font.setBold(True)
        label = QtWidgets.QLabel(name)
        label.setFont(label_font)
        header_layout.addWidget(label)
 
        # Push everything to the left
        header_layout.addItem(
            QtWidgets.QSpacerItem(
                0, 0,
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
        )
 
        stacked.addWidget(header_widget)
        stacked.addWidget(background)
        background.setMinimumHeight(int(header_layout.sizeHint().height() * 1.5))
 
    def mousePressEvent(self, *args):
        """Toggle expand/collapse when the header is clicked."""
        if self._content.isVisible():
            self.collapse()
        else:
            self.expand()
 
    def expand(self):
        """Show the content widget and switch to the down arrow."""
        self._content.setVisible(True)
        self._icon.setPixmap(self._expand_icon)
 
    def collapse(self):
        """Hide the content widget and switch to the right arrow."""
        self._content.setVisible(False)
        self._icon.setPixmap(self._collapse_icon)
 
 
class CollapsibleContainer(QtWidgets.QWidget):
    """
    A Maya-style collapsible section widget.
 
    Clicking the header bar shows or hides the content area, exactly
    like the collapsible sections in Maya's Attribute Editor and
    Tool Settings panels.
 
    Args:
        name (str):             Title shown in the header bar.
        collapsed (bool):       Start collapsed if True (default True).
        color_background (bool): Shade the content area slightly lighter,
                                 like Maya's grouped attribute sections.
 
    Usage:
        container = CollapsibleContainer("Maps Detected", collapsed=True)
        form = QtWidgets.QFormLayout(container.content_widget)
        form.addRow("File:", QtWidgets.QLabel("texture.png"))
        parent_layout.addWidget(container)
 
    Public methods (delegated from the header):
        container.expand()    — programmatically expand
        container.collapse()  — programmatically collapse
        container.toggle()    — toggle current state
    """
 
    def __init__(self, name, collapsed=True, color_background=False):
        super().__init__()
 
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
 
        # The content widget holds whatever the caller puts inside
        self._content_widget = QtWidgets.QWidget()
 
        if color_background:
            # Slightly lighter background — matches Maya's grouped sections
            self._content_widget.setStyleSheet(
                ".QWidget { background-color: rgb(73, 73, 73); "
                "margin-left: 2px; margin-right: 2px; }"
            )
 
        self._header = _CollapsibleHeader(name, self._content_widget)
        layout.addWidget(self._header)
        layout.addWidget(self._content_widget)
 
        # Delegate header methods so callers can control state externally
        self.collapse = self._header.collapse
        self.expand   = self._header.expand
        self.toggle   = self._header.mousePressEvent
 
        # Apply initial collapsed/expanded state
        if collapsed:
            self._header.collapse()
        else:
            self._header.expand()
 
    @property
    def content_widget(self):
        """The widget to add child content into."""
        return self._content_widget
 
 