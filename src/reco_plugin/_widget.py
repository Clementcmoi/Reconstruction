from typing import TYPE_CHECKING
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSpacerItem,
    QSizePolicy,
)

if TYPE_CHECKING:
    import napari

class ReconstructionWidget(QWidget):
    def __init__(self, napari_viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = napari_viewer
        self.setup_ui()
        self.connect_signals()

    def connect_signals(widget):
        """
        Connect signals for layer updates.
        """
        widget.viewer.layers.events.inserted.connect(lambda event: self.update_layer_selections(widget, event))
        widget.viewer.layers.events.removed.connect(lambda event: self.update_layer_selections(widget, event))
        widget.viewer.layers.events.changed.connect(lambda event: self.update_layer_selections(widget, event))
        widget.viewer.layers.events.reordered.connect(lambda event: self.update_layer_selections(widget, event))

    def update_layer_selections(widget, event=None):
        """
        Update the QComboBox selections with the list of layers.
        """
        layers = [layer.name for layer in widget.viewer.layers]
        for combobox in [widget.sample_selection]:
            combobox.clear()
            combobox.addItems(layers)

        if hasattr(widget, 'darkfield_checkbox') and widget.darkfield_checkbox.isChecked() and widget.darkfield_selection:
            widget.darkfield_selection.clear()
            widget.darkfield_selection.addItems(layers)
        
        if hasattr(widget, 'flatfield_checkbox') and widget.flatfield_checkbox.isChecked() and widget.flatfield_selection:
            widget.flatfield_selection.clear()
            widget.flatfield_selection.addItems(layers) 

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("LCS Processing"))