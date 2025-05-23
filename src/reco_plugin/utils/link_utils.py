# Imports
from cupyx.scipy.ndimage import shift
import numpy as np
import cupy as cp
from skimage.transform import resize
import h5py
import gc

# Local imports
from ..processing.cor import *
from ..processing.phase import unsharp_mask
from ..processing.reconstruction import (
    reconstruct_from_sinogram_slice, create_angles,
    create_disk_mask
)
from ..processing.angles import (
    find_angles_in_dataset, find_opposite_pairs_best_match,
    create_sinogram_slice_from_pairs, create_sinograms_from_pairs
)

def add_image_to_layer(results, img_name, viewer):
    for name, image in results.items():
        if isinstance(image, cp.ndarray):
            image = image.get()
        viewer.add_image(image.real, name=f"{name}_{img_name}")

def clear_memory(variables):
    for var in variables:
        del var
    gc.collect()
    cp._default_memory_pool.free_all_blocks()

def get_projections(viewer, prefix, slice_idx=None, fallback_func=None):
    for layer in viewer.layers:
        if layer.name.startswith(prefix):
            try:
                key = prefix 
                return {key: layer.data}
            except Exception as e:
                print(f"Error retrieving data from layer {layer.name}: {e}")
    return fallback_func() if fallback_func else None

def get_angles(viewer, experiment, shape, full=True):
    if viewer.layers[experiment.sample_images].metadata.get('paths') and hasattr(h5py, 'File'):
        with h5py.File(viewer.layers[experiment.sample_images].metadata['paths'][0], "r") as f:
            angles = np.radians(find_angles_in_dataset(f, shape))[0]
            if not full:
                angles = angles[:shape]
    else:
        angles = create_angles(np.empty((shape, shape)), end=2 * np.pi if full else np.pi)
    return angles

def apply_mask_and_reconstruct(sinogram, angles, sigma, coeff, apply_unsharp=False):
    mask = create_disk_mask(sinogram)
    slice_ = reconstruct_from_sinogram_slice(sinogram, angles) * mask
    if apply_unsharp:
        slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()
    return slice_

def convert_cor_to_shift(cor, width):
    shift = width // 2 - cor
    return shift

def pad_and_shift_projection(projs, cor):
    shift_value = convert_cor_to_shift(cor, projs.shape[1])
    pad_width = abs(int(shift_value))
    
    padded_projs = cp.pad(projs, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
    shifted = shift(padded_projs, (0, shift_value), order=1, mode='constant').get()

    return shifted

def resize_to_target(slice_, target_shape):
    if slice_.shape != target_shape:
        resized = resize(slice_, target_shape, mode='constant', anti_aliasing=True)
        return resized
    return slice_

def load_angles_and_create_sinograms(viewer, experiment, projs, cor):
    metadata_paths = viewer.layers[experiment.sample_images].metadata.get('paths', [])
    if not metadata_paths:
        raise ValueError("No paths found in metadata.")

    hdf5_path = metadata_paths[0]

    with h5py.File(hdf5_path, 'r') as hdf5_file:
        angles = np.radians(find_angles_in_dataset(hdf5_file, projs.shape[0])[0])
        pairs = find_opposite_pairs_best_match(angles)
        angles = angles[:pairs[-1][0] + 1]
        if projs.ndim != 2:
            sino = create_sinograms_from_pairs(projs, 2 * cor, pairs)
        else:
            sino = create_sinogram_slice_from_pairs(projs, 2 * cor, pairs)

    return sino, angles