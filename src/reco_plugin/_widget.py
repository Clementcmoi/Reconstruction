from typing import TYPE_CHECKING
from qtpy.QtWidgets import (
    QLabel, 
    QVBoxLayout, 
    QWidget,
    QSpacerItem, 
    QSizePolicy,
)

from .ui_sections import (
    add_sample_selection_section,
    add_preprocessing_section,
    add_paganin_section,
    add_double_flatfield_section,
    add_angles_section,
    add_center_of_rotation_section,
    add_process_section,
)
from .ui_mp_sections import (
    add_general_parameters_section,
    add_multi_paganin_sections,
)
from .utils.layer_utils import LayerUtils
from .utils.experiment import Experiment, MExperiment

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

        add_sample_selection_section(self)
        add_preprocessing_section(self)
        add_paganin_section(self)
        add_double_flatfield_section(self)
        add_angles_section(self)
        add_center_of_rotation_section(self)
        add_process_section(self)

        LayerUtils.update_layer_selections(self)
        self.layout().addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

class MultiPaganinWidget(QWidget):
    def __init__(self, napari_viewer: 'napari.Viewer'):
        super().__init__()
        self.viewer = napari_viewer
        self.experiment = MExperiment()
        self.setup_ui()
        LayerUtils.connect_signals(self) 

    def setup_ui(self):
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Multi-Paganin"))

        add_sample_selection_section(self)
        add_preprocessing_section(self)
        add_general_parameters_section(self)

        add_multi_paganin_sections(self)
        
        LayerUtils.update_layer_selections(self)
        