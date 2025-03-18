from qtpy.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel

def create_processing_dialog(parent, message="Processing..."):
    """
    Create and display a dialog with a message to indicate that processing is ongoing.
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle("Processing")
    layout = QVBoxLayout()
    label = QLabel(message)
    layout.addWidget(label)
    dialog.setLayout(layout)
    dialog.setFixedSize(200, 100)
    dialog.show()
    QApplication.processEvents()
    return dialog

def apply_corrections(viewer, experiment):
    """
    Apply flatfield and darkfield corrections to the sample layers 
    using the data stored in the experiment object.
    """
    print("Applying corrections")
    sample_layer = viewer.layers[experiment.sample_images].data

    if experiment.darkfield is not None:
        darkfield_layer = viewer.layers[experiment.darkfield].data
        sample_layer = sample_layer - darkfield_layer

    if experiment.flatfield is not None:
        flatfield_layer = viewer.layers[experiment.flatfield].data
        sample_layer = sample_layer / flatfield_layer

    return sample_layer

def add_image_to_layer(results, method, viewer):
    """
    Add the resulting image to the viewer as a new layer.
    """
    for name, image in results.items():
        viewer.add_image(image.real, name=f"{name}_{method}")

def process_one_slice(experiment, viewer):
    pass

def process_all_slices(experiment, viewer):
    pass