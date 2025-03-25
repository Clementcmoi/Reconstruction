from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import tifffile
import numpy as np
import os

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]

def write_tiff(path: str, data: Any, meta: dict) -> list[str]:
    """Write data to a TIFF file."""
    if not path.endswith('.tif'):
        name = os.path.splitext(path)[0]
        path = f"{name}.tif"

    tifffile.imwrite(path, data.astype(np.float32), imagej=True)
    return [path]