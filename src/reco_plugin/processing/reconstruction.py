import astra
import numpy as np
from skimage.draw import disk

from math import pi

def create_angles(sinogram: np.ndarray, end: float = pi) -> np.ndarray:
    """
    Create angles for a sinogram.
    """
    if sinogram.ndim == 2:
        num_angles = sinogram.shape[0]  # Nombre d'angles pour un sinogramme 2D
    elif sinogram.ndim == 3:
        num_angles = sinogram.shape[1]  # Nombre d'angles pour un sinogramme 3D
    else:
        raise ValueError("Sinogram must be 2D or 3D.")

    angles = np.linspace(0, end, num_angles, endpoint=False)
    return angles

def reconstruct_from_sinogram_slice(sinogram: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 2D image from a sinogram using FBP_CUDA algorithm from ASTRA Toolbox.
    """

    # Définition des géométries de projection et du volume
    proj_geom = astra.create_proj_geom('parallel', 1, sinogram.shape[1], angles)
    vol_geom = astra.create_vol_geom(sinogram.shape[1], sinogram.shape[1])

    # Création des objets de données pour le sinogramme et la reconstruction
    sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)
    rec_id = astra.data2d.create('-vol', vol_geom)

    # Configuration et exécution de l'algorithme FBP_CUDA
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Récupération et retour de la reconstruction
    reconstruction = astra.data2d.get(rec_id)

    # Libération des ressources ASTRA
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)

    return reconstruction * 12

def create_disk_mask(sinogram: np.ndarray) -> np.ndarray:
    """
    Create a circular disk mask for the sinogram.
    """
    if sinogram.ndim == 2:
        _, width = sinogram.shape
    elif sinogram.ndim == 3:
        _, width = sinogram.shape[1:]
        
    disk_mask = np.zeros((width, width))
    rr, cc = disk((width // 2, width // 2), (width // 2) - 1)
    disk_mask[rr, cc] = 1

    return disk_mask