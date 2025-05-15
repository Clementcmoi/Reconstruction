from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import tifffile
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QInputDialog

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]

def write_tiff(path: str, data: Any, meta: dict) -> list[str]:
    """Write data to a TIFF file."""
    if not path.endswith('.tif'):
        name = os.path.splitext(path)[0]
        path = f"{name}.tif"

    tifffile.imwrite(path, data)
    return [path]

def write_tiff_stack(path: str, data: Sequence[DataType], meta: dict) -> list[str]:
    """Write a volume to TIFF files slice by slice."""
    app_created = False
    if not QApplication.instance():
        app = QApplication([])
        app_created = True 
    max_index = len(data) - 1 
    start_slice, ok1 = QInputDialog.getInt(None, "Input", "Start slice (0-based index):", 0, 0, max_index, 1)
    if not ok1:
        raise ValueError("Slice range selection was cancelled.")
    end_slice, ok2 = QInputDialog.getInt(
        None, "Input", "End slice (0-based index):", max_index, start_slice, max_index, 1
    ) 
    if not ok2:
        raise ValueError("Slice range selection was cancelled.")
    if app_created:
        app.quit()  

    if not os.path.isdir(path):
        path = os.path.splitext(path)[0]  # Remove file extension
        os.makedirs(path, exist_ok=True)

    saved_files = []
    for i in range(start_slice, end_slice + 1):
        slice_path = os.path.join(path, f"slice_{i:04d}.tif")
        tifffile.imwrite(slice_path, data[i], imagej=True)
        saved_files.append(slice_path)

    return [path]
