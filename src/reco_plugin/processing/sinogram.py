import numpy as np
import cupy as cp
from tqdm import tqdm

def apply_left_weighting(CoR: int) -> cp.ndarray:
    """
    Generate linear weights for the left part of the projections.
    Handles both 2D and 3D projections.
    """
    weights = cp.linspace(0, 1, CoR)
    weights = weights[None, :]  # Expand for 2D
    return weights

def create_sinogram_slice(proj: cp.ndarray, CoR: int) -> np.ndarray:
    """
    Create a sinogram from a set of projections, applying left weighting.
    """
    print(f"Creating sinogram slice with CoR: {CoR}")
    weights = apply_left_weighting(CoR)

    theta, ny = proj.shape
    proj_copy = cp.copy(proj)  # Copy to avoid modifying the original data
    proj_copy[:, :CoR] *= weights

    sino = cp.zeros((theta // 2, 2 * ny - CoR))

    flip = proj_copy[:theta // 2, ::-1] 
    sino[:, :ny] += flip 
    sino[:, -ny:] += proj_copy[theta // 2:, :] 

    return sino.get()

def create_sinogram(projs: np.ndarray, CoR: int) -> np.ndarray:
    """
    Create sinograms from a set of projections, processing one slice at a time on the GPU.
    """
    sinos = []
    for i in tqdm(range(projs.shape[1]), desc="Creating sinograms"):

        sino = create_sinogram_slice(cp.asarray(projs[:, i, :]), CoR, i)
        sinos.append(sino)

    return np.array(sinos)  # Combine all sinograms into a numpy array