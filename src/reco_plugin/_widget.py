from typing import TYPE_CHECKING
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel, 
    QLineEdit, 
    QPushButton, 
    QVBoxLayout, 
    QHBoxLayout, 
    QWidget,
    QComboBox, 
    QCheckBox,
    QSpacerItem, 
    QSizePolicy,
    QInputDialog
)

from .widgets._section import *
from .widgets._utils import LayerUtils, Experiment

if TYPE_CHECKING:
    import napari

class ReconstructionWidget(QWidget):
    def __init__(self, napari_viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = napari_viewer
        self.experiment = Experiment()
        self.setup_ui()
        LayerUtils.connect_signals(self) 

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Reconstruction"))

        add_layer_and_slice_selection_section(self)
        add_darkfield_section(self)
        add_flatfield_section(self)

        add_paganin_section(self)

        add_double_flatfield_section(self)

        add_half_acquisition_section(self)

        add_process_section(self)

        LayerUtils.update_layer_selections(self)
        self.layout().addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )