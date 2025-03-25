try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import ReconstructionWidget
from ._writer import write_tiff

__all__ = (
    "__version__",
    "ReconstructionWidget", 
    "write_tiff")

def napari_experimental_provide_dock_widget():
    return ReconstructionWidget