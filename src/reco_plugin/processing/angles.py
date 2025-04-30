import numpy as np
import cupy as cp
import h5py
from tqdm import tqdm

from .sinogram import apply_left_weighting

def find_angles_in_dataset(file, nz, group=None, path="", results=None, start_tol=1, end_tol=1):
    """
    Find all 1D datasets of length `nz`, starting at ~0 and ending at ~360.
    Returns the datasets and their corresponding values.
    """
    if results is None:
        results = []

    if group is None:
        group = file

    for key in group:
        item = group[key]
        current_path = f"{path}/{key}"
        
        if isinstance(item, h5py.Group):
            find_angles_in_dataset(
                file, nz, group=item, path=current_path, results=results,
                start_tol=start_tol, end_tol=end_tol
            )
        elif isinstance(item, h5py.Dataset):
            if item.ndim == 1 and item.shape[0] == nz:
                try:
                    data = item[()]
                    if (
                        abs(data[0] - 0) <= start_tol and
                        abs(data[-1] - 360) <= end_tol
                    ):
                        results.append(data)
                        print(f"Dataset found: {current_path} with {data.shape[0]} values")
                except Exception as e:
                    print(f"Error reading {current_path}: {e}")
    return results

def find_opposite_pairs_best_match(angles):
    """
    Find the best matching opposite pairs of angles.
    """
    angles = np.array(angles) % (2 * np.pi)
    N = len(angles)
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    
    pairs = []
    used = set()

    for i in range(N):
        if i in used:
            continue

        ref_angle = (angles[i] + np.pi) % (2 * np.pi)
        # Find the index of the closest angle to ref_angle
        j = np.argmin(np.abs((sorted_angles - ref_angle + np.pi) % (2 * np.pi) - np.pi))
        j_global = sorted_indices[j]

        if j_global != i and j_global not in used:
            pairs.append((i, j_global))
            used.add(i)
            used.add(j_global)

    return pairs

def create_sinogram_slice_from_pairs(proj: cp.ndarray, CoR: int, angle_pairs: list[tuple[int, int]]) -> np.ndarray:
    """
    Create a sinogram using matched pairs of angles, applying left weighting to both projections.
    Since the first projection is flipped, the weighting becomes symmetrical.
    """
    weights = apply_left_weighting(CoR)
    ny = proj.shape[1]
    sino = cp.zeros((len(angle_pairs), 2 * ny - CoR))

    for k, (i, j) in enumerate(angle_pairs):
        # Projection i (left, flipped)
        proj_i = cp.copy(proj[i])
        proj_i[:CoR] *= weights[0]
        flip_i = proj_i[::-1]

        # Projection j (right), same weighting
        proj_j = cp.copy(proj[j])
        proj_j[:CoR] *= weights[0]

        # Assemble into sinogram
        sino[k, :ny] += flip_i
        sino[k, -ny:] += proj_j

    return sino.get()

def create_sinograms_from_pairs(projs: np.ndarray, CoR: int, angle_pairs: list[tuple[int, int]]) -> np.ndarray:
    """
    Create sinograms for an entire stack of slices using matched pairs of angles.
    Applies left weighting to both projections.
    
    Parameters:
    - proj: np.ndarray of shape (n_angles, n_slices, n_pixels)
    - CoR: center of rotation
    - angle_pairs: list of (i, j) pairs of angles
    
    Returns:
    - sino: np.ndarray of shape (n_slices, len(angle_pairs), 2 * n_pixels - CoR)
    """
    weights = apply_left_weighting(CoR).get()
    n_angles, n_slices, n_pixels = projs.shape
    sino = np.zeros((n_slices, len(angle_pairs), 2 * n_pixels - CoR))

    for slice_idx in tqdm(range(n_slices), desc="Creating sinograms"):
        for k, (i, j) in enumerate(angle_pairs):
            # Projection i (left, flipped)
            proj_i = np.copy(projs[i, slice_idx])
            proj_i[:CoR] *= weights[0]
            flip_i = proj_i[::-1]

            # Projection j (right), same weighting
            proj_j = np.copy(projs[j, slice_idx])
            proj_j[:CoR] *= weights[0]

            # Assemble into sinogram
            sino[slice_idx, k, :n_pixels] += flip_i
            sino[slice_idx, k, -n_pixels:] += proj_j

    return sino
