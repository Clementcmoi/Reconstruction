def process_try_center_of_rotation(self, widget):
    """
    Process the center of rotation based on the widget values.
    """
    print(f"Processing Center of Rotation")
    try:
        self.center_of_rotation = int(widget.center_of_rotation.value())
        print(f"Center of Rotation: {self.center_of_rotation}")
    except Exception as e:
        print(f"Error processing center of rotation: {e}")

def process_precise_local(self, widget):
    """
    Process the precise local based on the widget values.
    """
    print(f"Processing Precise Local")
    try:
        self.precise_local = int(widget.precise_local.value())
        print(f"Precise Local: {self.precise_local}")
    except Exception as e:
        print(f"Error processing precise local: {e}")