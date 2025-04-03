from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel, 
    QLineEdit, 
    QVBoxLayout, 
    QHBoxLayout, 
    QComboBox, 
    QCheckBox,
    QSpinBox,
    QPushButton,
    QGroupBox,
)

from ._processing import process_try_paganin, process_all_slices

def update_slice_range(widget):
    """
    Adjusts the range of the `slice_selector` according to the selected layer.
    """
    selected_layer_name = widget.sample_selection.currentText()
    if (selected_layer_name in widget.viewer.layers):
        selected_layer = widget.viewer.layers[selected_layer_name]
        if hasattr(selected_layer, 'data'):
            max_slices = selected_layer.data.shape[0] - 1
            widget.slice_selector.setMaximum(max_slices)

def add_sample_selection_section(widget):
    """
    Add sample and slice selection section to the widget.
    """
    group_box = QGroupBox("Sample Selection")
    layout = QVBoxLayout()

    # Sample selection
    layout_sample = QHBoxLayout()
    sample_label = QLabel("Sample:")
    widget.sample_selection = QComboBox()
    layout_sample.addWidget(sample_label)
    layout_sample.addWidget(widget.sample_selection)

    # Slice selection
    layout_slice = QHBoxLayout()
    slice_label = QLabel("Slice:")
    widget.slice_selection = QSpinBox()
    widget.slice_selection.setMinimum(0)
    widget.slice_selection.setMaximum(1000)
    layout_slice.addWidget(slice_label)
    layout_slice.addWidget(widget.slice_selection)

    layout.addLayout(layout_sample)
    layout.addLayout(layout_slice)

    group_box.setLayout(layout)

    widget.sample_selection.currentIndexChanged.connect(lambda: update_slice_range(widget))

    widget.layout().addWidget(group_box)

def toggle_field_widgets(widget, checked, layout, label_attr, selection_attr, label_text):
    if checked == Qt.Checked:
        if not getattr(widget, label_attr):
            setattr(widget, label_attr, QLabel(label_text))
        if not getattr(widget, selection_attr):
            combobox = QComboBox()
            combobox.addItems([layer.name for layer in widget.viewer.layers])
            setattr(widget, selection_attr, combobox)
        
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(getattr(widget, label_attr))
        horizontal_layout.addWidget(getattr(widget, selection_attr))
        layout.addLayout(horizontal_layout)
    else:
        widget_label = getattr(widget, label_attr)
        widget_selection = getattr(widget, selection_attr)
        if widget_label:
            layout.removeWidget(widget_label)
            widget_label.deleteLater()
            setattr(widget, label_attr, None)
        if widget_selection.isVisible():
            layout.removeWidget(widget_selection)
            widget_selection.deleteLater()
            setattr(widget, selection_attr, None)
    widget.layout().update()

def add_darkfield_section(widget):
    """
    Create and return the darkfield-related UI components.
    """
    darkfield_layout = QVBoxLayout()

    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Darkfield:"))
    widget.darkfield_checkbox = QCheckBox()
    widget.darkfield_checkbox.stateChanged.connect(
        lambda state: toggle_field_widgets(widget, state, darkfield_layout, 'darkfield_label', 'darkfield_selection', 'Select darkfield:')
    )
    checkbox_layout.addWidget(widget.darkfield_checkbox)

    darkfield_layout.addLayout(checkbox_layout)

    widget.darkfield_label = None
    widget.darkfield_selection = None

    return darkfield_layout


def add_flatfield_section(widget):
    """
    Create and return the flatfield-related UI components.
    """
    flatfield_layout = QVBoxLayout()

    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Flatfield:"))
    widget.flatfield_checkbox = QCheckBox()
    widget.flatfield_checkbox.stateChanged.connect(
        lambda state: toggle_field_widgets(widget, state, flatfield_layout, 'flatfield_label', 'flatfield_selection', 'Select flatfield:')
    )
    checkbox_layout.addWidget(widget.flatfield_checkbox)

    flatfield_layout.addLayout(checkbox_layout)

    widget.flatfield_label = None
    widget.flatfield_selection = None

    return flatfield_layout


def add_preprocessing_section(widget):
    """
    Add preprocessing section to the widget.
    """
    group_box = QGroupBox("Preprocessing")
    layout = QVBoxLayout() 

    flatfield_layout = add_flatfield_section(widget)
    layout.addLayout(flatfield_layout)

    darkfield_layout = add_darkfield_section(widget)
    layout.addLayout(darkfield_layout)

    apply_button = QPushButton("Apply")
    apply_button.clicked.connect(lambda: process_try_paganin(widget.experiment, widget.viewer))
    layout.addWidget(apply_button)

    group_box.setLayout(layout)

    widget.layout().addWidget(group_box)

def add_energy_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Energy (keV):"))
    widget.energy_input = QLineEdit()
    widget.energy_input.setText(str(widget.experiment.energy) if widget.experiment.energy is not None else "")
    layout.addWidget(widget.energy_input)
    widget.variables_layout.addLayout(layout)

def add_pixel_size_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Pixel size (m):"))
    widget.pixel_size_input = QLineEdit()
    widget.pixel_size_input.setText(str(widget.experiment.pixel) if widget.experiment.pixel is not None else "")
    layout.addWidget(widget.pixel_size_input)
    widget.variables_layout.addLayout(layout)

def add_effective_pixel_size_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Effective pixel size (m):"))
    widget.effective_pixel_size_input = QLineEdit()
    widget.effective_pixel_size_input.setText(str(widget.experiment.effective_pixel) if widget.experiment.effective_pixel is not None else "")
    layout.addWidget(widget.effective_pixel_size_input)
    widget.variables_layout.addLayout(layout)

def add_delta_beta_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Delta Beta Ratio:"))
    widget.db_input = QLineEdit()
    widget.db_input.setText(str(widget.experiment.db) if widget.experiment.db is not None else "")
    layout.addWidget(widget.db_input)
    widget.variables_layout.addLayout(layout)

def add_distance_object_detector_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Distance object-detector (m):"))
    widget.distance_object_detector_input = QLineEdit()
    widget.distance_object_detector_input.setText(str(widget.experiment.dist_object_detector) if widget.experiment.dist_object_detector is not None else "")
    layout.addWidget(widget.distance_object_detector_input)
    widget.variables_layout.addLayout(layout)

def add_unsharp_mask_layout(widget):
    layout = QHBoxLayout()
    layout.addWidget(QLabel("Unsharp mask → Sigma:"))
    widget.sigma_input = QLineEdit()
    widget.sigma_input.setText(str(widget.experiment.sigma) if widget.experiment.sigma is not None else "")
    layout.addWidget(widget.sigma_input)
    layout.addWidget(QLabel("Coefficient:"))
    widget.coeff_input = QLineEdit()
    widget.coeff_input.setText(str(widget.experiment.coeff) if widget.experiment.coeff is not None else "")
    layout.addWidget(widget.coeff_input)
    widget.variables_layout.addLayout(layout)

def toggle_paganin(widget, checked, layout):
    """
    Toggle the visibility of the energy input field and other parameters based on the Paganin checkbox state.
    """
    if checked == Qt.Checked:
        if not hasattr(widget, 'variables_layout'):
            widget.variables_layout = QVBoxLayout()
            layout.addLayout(widget.variables_layout)

        add_energy_layout(widget)
        add_pixel_size_layout(widget)
        add_effective_pixel_size_layout(widget)
        add_distance_object_detector_layout(widget)
        add_delta_beta_layout(widget)
        add_unsharp_mask_layout(widget)

        widget.paganin_apply_button = QPushButton("Apply")
        widget.paganin_apply_button.clicked.connect(lambda: process_try_paganin(widget.experiment, widget.viewer))
        layout.addWidget(widget.paganin_apply_button)

    else:
        if hasattr(widget, 'variables_layout') and widget.variables_layout:
            while widget.variables_layout.count():
                child = widget.variables_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            del widget.variables_layout

        if hasattr(widget, 'paganin_apply_button') and widget.paganin_apply_button:
            layout.removeWidget(widget.paganin_apply_button)
            widget.paganin_apply_button.deleteLater()
            del widget.paganin_apply_button

def add_paganin_section(widget):
    """
    Add Paganin section to the widget.
    """
    group_box = QGroupBox("Paganin")
    layout = QVBoxLayout()

    # Add Paganin checkbox
    widget.paganin_checkbox = QCheckBox("Enable Paganin")
    widget.paganin_checkbox.stateChanged.connect(
        lambda state: toggle_paganin(widget, state, layout)
    )
    layout.addWidget(widget.paganin_checkbox)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

def add_double_flatfield_section(widget):
    """
    Add double flatfield section to the widget.
    """
    group_box = QGroupBox("Double Flatfield")
    layout = QVBoxLayout()

    widget.double_flatfield_checkbox = QCheckBox("Enable Double Flatfield")
    layout.addWidget(widget.double_flatfield_checkbox)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)

def toggle_center_of_rotation(widget, state, layout):
    """
    Toggle the visibility of the center of rotation input field based on the acquisition type.
    """
    if state == 0:  # Standard Acquisition
        if hasattr(widget, 'cor_label') and widget.cor_label is not None:
            layout.removeWidget(widget.cor_label)
            widget.cor_label.deleteLater()
            widget.cor_label = None

        if hasattr(widget, 'cor_selection') and widget.cor_selection is not None:
            layout.removeWidget(widget.cor_selection)
            widget.cor_selection.deleteLater()
            widget.cor_selection = None

    else:  # Half Acquisition
        if not hasattr(widget, 'cor_label') or widget.cor_label is None:
            widget.cor_label = QLabel("Center of Rotation:")
            layout.addWidget(widget.cor_label)

        if not hasattr(widget, 'cor_selection') or widget.cor_selection is None:
            widget.cor_selection = QLineEdit()
            layout.addWidget(widget.cor_selection)

def add_center_of_rotation_section(widget):
    """
    Add center of reotation section to widget.
    """
    group_box = QGroupBox("Center of Rotation")
    layout = QVBoxLayout()

    acquisition_type_layout = QHBoxLayout()
    acquisition_type_label = QLabel("Acquisition Type:")
    widget.acquisition_type_selection = QComboBox()
    widget.acquisition_type_selection.addItems(["Standard Acquisition", "Half Acquisition"])
    acquisition_type_layout.addWidget(acquisition_type_label)
    acquisition_type_layout.addWidget(widget.acquisition_type_selection)
    widget.acquisition_type_selection.currentIndexChanged.connect(
        lambda state: toggle_center_of_rotation(widget, state, layout)
    )


    layout.addLayout(acquisition_type_layout)

    group_box.setLayout(layout)
    widget.layout().addWidget(group_box)