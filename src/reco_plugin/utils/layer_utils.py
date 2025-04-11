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
