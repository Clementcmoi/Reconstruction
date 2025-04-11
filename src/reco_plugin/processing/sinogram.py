import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def apply_left_weighting(projs: np.ndarray, CoR: int) -> np.ndarray:
    """
    Apply a linear weight to the left part of the projections.
    Handles both 2D and 3D projections.
    """
    weights = np.linspace(0, 1, CoR)
    if projs.ndim == 3:
        weights = weights[None, None, :]  # Expand for 3D
        projs[:, :, :CoR] *= weights
    elif projs.ndim == 2:
        weights = weights[None, :]  # Expand for 2D
        projs[:, :CoR] *= weights

    return projs 

def create_sinogram_slice(projs: np.ndarray, CoR: int, slice_idx: int) -> np.ndarray:
    """
    Create a sinogram from a set of projections.
    """
    if projs.ndim == 3:
        theta, _, ny = projs.shape

    elif projs.ndim == 2:
        theta, ny = projs.shape

    sino = np.zeros((theta//2, 2 * ny - CoR))

    flip = projs[:theta // 2, slice_idx, ::-1]  # np.flip optimisé

    sino[:, :ny] += flip
    sino[:,  -ny:] += projs[theta//2:, slice_idx, :]

    return sino

def create_sinogram(projs: np.ndarray, CoR: int) -> np.ndarray:
    """
    Create sinograms from a set of projections.
    """

    projs_weighted = apply_left_weighting(projs, CoR)

    sinos = np.array(
        Parallel(n_jobs=-1, backend='threading')(
            delayed(create_sinogram_slice)(projs_weighted, CoR, slice_idx)
            for slice_idx in tqdm(range(projs.shape[1]), desc='Creating sinograms')
        )
    )

    return sinos