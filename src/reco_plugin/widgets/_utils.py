from numpy import pi
from qtpy.QtCore import QSettings

class LayerUtils:
    @staticmethod
    def connect_signals(widget):
        """
        Connect signals for layer updates.
        """
        widget.viewer.layers.events.inserted.connect(lambda event: LayerUtils.update_layer_selections(widget, event))
        widget.viewer.layers.events.removed.connect(lambda event: LayerUtils.update_layer_selections(widget, event))
        widget.viewer.layers.events.changed.connect(lambda event: LayerUtils.update_layer_selections(widget, event))
        widget.viewer.layers.events.reordered.connect(lambda event: LayerUtils.update_layer_selections(widget, event))

    @staticmethod
    def update_layer_selections(widget, event=None):
        """
        Update the QComboBox selections with the list of layers.
        """
        layers = [layer.name for layer in widget.viewer.layers]
        widget.sample_selection.clear()
        widget.sample_selection.addItems(layers)

        if hasattr(widget, 'darkfield_checkbox') and widget.darkfield_checkbox.isChecked() and widget.darkfield_selection:
            widget.darkfield_selection.clear()
            widget.darkfield_selection.addItems(layers)
        
        if hasattr(widget, 'flatfield_checkbox') and widget.flatfield_checkbox.isChecked() and widget.flatfield_selection:
            widget.flatfield_selection.clear()
            widget.flatfield_selection.addItems(layers) 

class Experiment:
    def __init__(self): 
        self.settings = QSettings("Reco", "recoconfig")

        self.sample_images = None
        self.darkfield = None
        self.flatfield = None

        self.energy = None
        self.pixel = None
        self.delta = None
        self.beta = None
        self.dist_object_detector = None    

        self.double_flatfield = None

        self.center_of_rotation = None

        self.load_settings()
    
    def load_settings(self):
        for attr in vars(self):
            if attr not in ["settings"]:
                value = self.settings.value(attr, getattr(self, attr))
                setattr(self, attr, value)

    def save_settings(self):
        for attr in vars(self):
            if attr not in ["settings"]:
                self.settings.setValue(attr, getattr(self, attr))

        print(f"Saved Parameters: {vars(self)}")

    def update_parameters(self, widget):
        """
        Update the parameters based on the widget values.
        """
        print(f"Updating Parameters")
        try:
            self.sample_images = widget.sample_selection.currentText()

            self.darkfield = widget.darkfield_selection.currentText() if widget.darkfield_checkbox.isChecked() else None
            self.flatfield = widget.flatfield_selection.currentText() if widget.flatfield_checkbox.isChecked() else None

            self.energy = float(widget.energy_input.text())
            self.pixel = float(widget.pixel_size_input.text())
            self.delta = float(widget.delta_input.text())
            self.beta = float(widget.beta_input.text())
            self.dist_object_detector = float(widget.distance_object_detector_input.text())           
            self.double_flatfield = widget.double_flatfield_checkbox.isChecked()
            self.center_of_rotation = float(widget.center_of_rotation_input.text())

            self.save_settings()

        except ValueError as e:
            print(f"Error updating parameters: {e}")
