from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel, 
    QLineEdit, 
    QVBoxLayout, 
    QHBoxLayout, 
    QComboBox, 
    QCheckBox,
    QSpinBox,
    QPushButton
)

from ._processing import process_try_paganin, process_all_slices

def add_layer_and_slice_selection_section(widget):
        """
        Ajoute la sélection du layer et du numéro de slice.
        """
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Select sample:"))
        widget.sample_selection = QComboBox()
        layout.addWidget(widget.sample_selection)

        layout.addWidget(QLabel("Radio:"))
        widget.slice_selector = QSpinBox()
        widget.slice_selector.setMinimum(0) 
        widget.slice_selector.setMaximum(1000) 
        layout.addWidget(widget.slice_selector)

        widget.layout().addLayout(layout) 

        widget.sample_selection.currentIndexChanged.connect(lambda: update_slice_range(widget))

def update_slice_range(widget):
    """
    Ajuste la plage du `slice_selector` en fonction du layer sélectionné.
    """
    selected_layer_name = widget.sample_selection.currentText()
    if (selected_layer_name in widget.viewer.layers):
        selected_layer = widget.viewer.layers[selected_layer_name]
        if hasattr(selected_layer, 'data'):
            max_slices = selected_layer.data.shape[0] - 1
            widget.slice_selector.setMaximum(max_slices)

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
    Add darkfield-related UI components.
    """
    darkfield_layout = QVBoxLayout()

    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Darkfield:"))
    widget.darkfield_checkbox = QCheckBox()
    widget.darkfield_checkbox.stateChanged.connect(lambda state: toggle_field_widgets(widget, state, darkfield_layout, 'darkfield_label', 'darkfield_selection', 'Select darkfield:'))
    checkbox_layout.addWidget(widget.darkfield_checkbox)

    darkfield_layout.addLayout(checkbox_layout)
    widget.layout().addLayout(darkfield_layout)

    widget.darkfield_label = None
    widget.darkfield_selection = None

def add_flatfield_section(widget):
    """
    Add flatfield-related UI components.
    """
    flatfield_layout = QVBoxLayout()

    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Flatfield:"))
    widget.flatfield_checkbox = QCheckBox()
    widget.flatfield_checkbox.stateChanged.connect(lambda state: toggle_field_widgets(widget, state, flatfield_layout, 'flatfield_label', 'flatfield_selection', 'Select flatfield:'))
    checkbox_layout.addWidget(widget.flatfield_checkbox)

    flatfield_layout.addLayout(checkbox_layout)
    widget.layout().addLayout(flatfield_layout)

    widget.flatfield_label = None
    widget.flatfield_selection = None


def ensure_variables_layout(widget):
    """
    Checks whether 'widget' has a layout named 'variables_layout'. 
    If not, it creates it and adds it to the widget's main layout.
    Then returns this layout.
    """
    if not hasattr(widget, 'variables_layout'):
        widget.variables_layout = QVBoxLayout()
        widget.layout().addLayout(widget.variables_layout)

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

def add_delta_and_beta_layout(widget):
    layout = QHBoxLayout()
    
    layout.addWidget(QLabel("Delta:"))
    widget.delta_input = QLineEdit()
    widget.delta_input.setText(str(widget.experiment.delta) if widget.experiment.delta is not None else "")
    widget.delta_input.textChanged.connect(lambda: update_delta_beta_ratio(widget))
    layout.addWidget(widget.delta_input)
    
    layout.addWidget(QLabel("Beta:"))
    widget.beta_input = QLineEdit()
    widget.beta_input.setText(str(widget.experiment.beta) if widget.experiment.beta is not None else "")
    widget.beta_input.textChanged.connect(lambda: update_delta_beta_ratio(widget))
    layout.addWidget(widget.beta_input)
    
    widget.variables_layout.addLayout(layout)

    # Add delta/beta ratio
    ratio_layout = QHBoxLayout()
    ratio_layout.addWidget(QLabel("Delta/Beta:"))
    widget.delta_beta_ratio_label = QLabel()
    widget.delta_beta_ratio_label.setText(
        str(float(widget.experiment.delta) / float(widget.experiment.beta))
        if widget.experiment.delta is not None and widget.experiment.beta is not None and float(widget.experiment.beta) != 0
        else "N/A"
    )
    ratio_layout.addWidget(widget.delta_beta_ratio_label)
    widget.variables_layout.addLayout(ratio_layout)

def update_delta_beta_ratio(widget):
    """
    Update the delta/beta ratio label whenever delta or beta is changed.
    """
    try:
        delta = float(widget.delta_input.text())
        beta = float(widget.beta_input.text())
        if beta != 0:
            widget.delta_beta_ratio_label.setText(f"{delta / beta:.4f}")
        else:
            widget.delta_beta_ratio_label.setText("N/A")
    except ValueError:
        widget.delta_beta_ratio_label.setText("N/A")

def add_distance_object_detector_layout(widget):

    layout = QHBoxLayout()

    layout.addWidget(QLabel("Distance object-detector (m):"))
    widget.distance_object_detector_input = QLineEdit()
    widget.distance_object_detector_input.setText(str(widget.experiment.dist_object_detector) if widget.experiment.dist_object_detector is not None else "")
    layout.addWidget(widget.distance_object_detector_input)

    widget.variables_layout.addLayout(layout)

def update_parameters(widget):
    """
    Update the parameters dictionary based on widget values.
    """
    widget.experiment.update_parameters(widget)

def add_paganin_section(widget):
    """
    Add Paganin-related UI components.
    """
    ensure_variables_layout(widget)

    widget.variables_layout.addWidget(QLabel("Paganin Parameters"))

    add_energy_layout(widget)
    add_pixel_size_layout(widget)
    add_effective_pixel_size_layout(widget)
    add_delta_and_beta_layout(widget)
    add_distance_object_detector_layout(widget)

def add_double_flatfield_section(widget):
    """
    Ajoute une case à cocher pour activer/désactiver la double correction flatfield.
    """
    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Double flatfield:"))
    
    widget.double_flatfield_checkbox = QCheckBox()
    checkbox_layout.addWidget(widget.double_flatfield_checkbox)

    widget.layout().addLayout(checkbox_layout)

def add_half_acquisition_section(widget):
    """
    Ajoute une case à cocher pour activer/désactiver la demi-acquisition.
    """
    checkbox_layout = QHBoxLayout()
    checkbox_layout.addWidget(QLabel("Half acquisition:"))
    
    widget.half_acquisition_checkbox = QCheckBox()
    widget.half_acquisition_checkbox.stateChanged.connect(lambda state: toggle_center_of_rotation(widget, state))
    checkbox_layout.addWidget(widget.half_acquisition_checkbox)

    widget.layout().addLayout(checkbox_layout)

    widget.center_of_rotation_layout = QHBoxLayout()
    widget.center_of_rotation_label = QLabel("Center of rotation:")
    widget.center_of_rotation_input = QLineEdit()
    widget.center_of_rotation_input.setText(str(widget.experiment.center_of_rotation) if widget.experiment.center_of_rotation is not None else "")
    widget.center_of_rotation_label.hide()
    widget.center_of_rotation_input.hide()
    widget.center_of_rotation_layout.addWidget(widget.center_of_rotation_label)
    widget.center_of_rotation_layout.addWidget(widget.center_of_rotation_input)
    widget.layout().addLayout(widget.center_of_rotation_layout)

def toggle_center_of_rotation(widget, checked):
    """
    Toggle the center of rotation input field.
    """
    if checked == Qt.Checked:
        widget.center_of_rotation_label.show()
        widget.center_of_rotation_input.show()
    else:
        widget.center_of_rotation_label.hide()
        widget.center_of_rotation_input.hide()

def add_process_section(widget):
    """
    Ajoute un bouton pour lancer le processus de reconstruction sur une slice et un bouton pour le lancer sur toutes les slices.
    """

    process_layout = QHBoxLayout()

    widget.process_slice_button = QPushButton("Try Paganin")
    widget.process_slice_button.clicked.connect(lambda: process_slice(widget))
    process_layout.addWidget(widget.process_slice_button)

    widget.process_all_button = QPushButton("Process all")
    widget.process_all_button.clicked.connect(lambda: process_all(widget))
    process_layout.addWidget(widget.process_all_button)

    widget.layout().addLayout(process_layout)

def process_slice(widget):
    """
    Process the current slice.
    """
    print("Processing slice")
    widget.experiment.update_parameters(widget)
    result = process_try_paganin(widget.experiment, widget.viewer)

def process_all(widget):
    """
    Process all slices.
    """
    print("Processing all slices")
    widget.experiment.update_parameters(widget)
    result = process_all_slices(widget.experiment, widget.viewer)






