from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication

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