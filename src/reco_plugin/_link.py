# Imports
from cupyx.scipy.ndimage import shift
import numpy as np
import cupy as cp
from tqdm import tqdm
from skimage.transform import resize
import h5py
import gc

# Local imports
from .processing.cor import *
from .processing.phase import paganin_filter, unsharp_mask, paganin_filter_slice
from .processing.process import apply_flat_darkfield, double_flatfield_correction
from .processing.reconstruction import (
    reconstruct_from_sinogram_slice, create_angles,
    create_disk_mask
)
from .processing.sinogram import create_sinogram, create_sinogram_slice
from .processing.angles import (
    find_angles_in_dataset, find_opposite_pairs_best_match,
    create_sinogram_slice_from_pairs, create_sinograms_from_pairs
)
from .utils.qt_helpers import create_processing_dialog, PlotWindow



# Utility functions
def add_image_to_layer(results, img_name, viewer):
    for name, image in results.items():
        # Explicitly convert CuPy arrays to NumPy arrays before adding to the viewer
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
                key = prefix  # Dynamically set the key based on the prefix
                return {key: layer.data}  # Use dynamic key
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

def pad_and_shift_projection(projs, cor):
    pad_width = abs(cor)
    padded_projs = cp.pad(projs, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
    shifted = shift(padded_projs, (0, cor), order=1, mode='constant').get()
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

# Main processing functions
def call_preprocess(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=["sample_images", "darkfield", "flatfield", "bigdata"])

        sample = np.transpose(viewer.layers[experiment.sample_images].data, viewer.dims.order)
        dark = np.mean(np.transpose(viewer.layers[experiment.darkfield].data, viewer.dims.order), axis=0) if experiment.darkfield else None
        flat = np.mean(np.transpose(viewer.layers[experiment.flatfield].data, viewer.dims.order), axis=0) if experiment.flatfield else None

        corrected = apply_flat_darkfield(sample, flat, dark)
        if not experiment.bigdata:
            add_image_to_layer(corrected, experiment.sample_images, viewer)

        clear_memory([sample, dark, flat])
        return corrected
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    finally:
        dialog.close()


def call_paganin(experiment, viewer, widget, one_slice=False):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "pixel", "effective_pixel", "dist_object_detector", "energy", "db", "sigma", "coeff"
        ])

        projs = get_projections(viewer, 'preprocess', slice_idx=int(experiment.slice_idx) if one_slice else None,
                                fallback_func=lambda: call_preprocess(experiment, viewer, widget))['preprocess']

        if not widget.paganin_checkbox.isChecked():
            return {'paganin': projs}

        if one_slice:
            result = paganin_filter_slice(
                projs, int(experiment.slice_idx), float(experiment.energy), float(experiment.pixel),
                float(experiment.effective_pixel), float(experiment.dist_object_detector), float(experiment.db)
            )
        else:
            result = paganin_filter(
                projs, float(experiment.energy), float(experiment.pixel), float(experiment.effective_pixel),
                float(experiment.dist_object_detector), float(experiment.db)
            )

        if not experiment.bigdata:
            add_image_to_layer(result, experiment.sample_images, viewer)

        clear_memory([projs])
        return result
    except Exception as e:
        print(f"Error during Paganin: {e}")
    finally:
        dialog.close()

def call_standard_cor_test(experiment, viewer, widget):
    print("Starting Standard COR test.")
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        # Mise à jour des paramètres de base
        base_parameters = ["slice_idx", "cor_min", "cor_max", "cor_step", "double_flatfield"]
        experiment.update_parameters(widget, parameters_to_update=base_parameters)

        slice_idx = int(experiment.slice_idx)
        sigma, coeff = (float(experiment.sigma), float(experiment.coeff)) if widget.paganin_checkbox.isChecked() else (0, 0)
        cor_range = np.arange(int(experiment.cor_min), int(experiment.cor_max) + int(experiment.cor_step), int(experiment.cor_step))

        # Vérification si Paganin doit être appliqué
        if widget.paganin_checkbox.isChecked():
            projs = get_projections(viewer, 'paganin', slice_idx=slice_idx, fallback_func=lambda: call_paganin(experiment, viewer, widget, one_slice=True))['paganin']
            apply_unsharp = True
        else:
            projs = get_projections(viewer, 'preprocess', slice_idx=slice_idx, fallback_func=lambda: call_preprocess(experiment, viewer, widget))['preprocess']
            apply_unsharp = False

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']

        projs = cp.asarray(projs)
        target_shape, slices = None, []

        for cor in tqdm(cor_range, desc="Generating Slices"):
            sino = pad_and_shift_projection(projs, cor)
            angles = get_angles(viewer, experiment, sino.shape[0]) if widget.angles_checkbox.isChecked() else create_angles(sino, end=2 * np.pi)
            slice_ = apply_mask_and_reconstruct(sino, angles, sigma, coeff, apply_unsharp=apply_unsharp)
            if target_shape is None:
                target_shape = slice_.shape
            slices.append(resize_to_target(slice_, target_shape))

        result = {'slice': np.array(slices)}
        if not experiment.bigdata:
            add_image_to_layer(result, "cor_test", viewer)
        clear_memory([projs])
        return result
    except Exception as e:
        print(f"#Error during Standard COR test: {e}")
    finally:
        dialog.close()


def call_find_global_cor(experiment, viewer, widget):
    print("Starting global COR calculation.")
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=["slice_idx", "center_of_rotation", "acquisition_type", "double_flatfield"])
        projs = get_projections(viewer, 'paganin', fallback_func=lambda: call_paganin(experiment, viewer, widget))['paganin']
        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']

        cor, plot_data = calc_cor(projs)
        cor = cor[np.isfinite(cor)]
        cor_mean = np.mean(cor[(cor > np.mean(cor) - np.std(cor)) & (cor < np.mean(cor) + np.std(cor))])
        widget.center_of_rotation_input.setText(str(cor_mean))

        widget.plot_window = PlotWindow(plot_data, cor_values=cor)
        widget.plot_window.show()
    except Exception as e:
        print(f"Error during global COR calculation: {e}")
    finally:
        dialog.close()


def call_half_cor_test(experiment, viewer, widget):
    print("Starting half COR test.")
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=["slice_idx", "double_flatfield", "center_of_rotation", "cor_fenetre"])
        slice_idx = int(experiment.slice_idx)
        sigma, coeff = float(experiment.sigma), float(experiment.coeff)
        cor_test, cor_fenetre = int(experiment.center_of_rotation), int(experiment.cor_fenetre)
        cor_range = np.arange(cor_test - cor_fenetre, cor_test + cor_fenetre)

        if widget.paganin_checkbox.isChecked():
            projs = get_projections(viewer, 'paganin', slice_idx=slice_idx, fallback_func=lambda: call_paganin(experiment, viewer, widget, one_slice=True))['paganin']
            apply_unsharp = True
        else:
            projs = get_projections(viewer, 'preprocess', slice_idx=slice_idx, fallback_func=lambda: call_preprocess(experiment, viewer, widget))['preprocess']
            apply_unsharp = False

        if projs.ndim != 2:
            projs = projs[:, slice_idx]

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']

        projs = cp.asarray(projs)
        target_shape, slices = None, []

        for cor in tqdm(cor_range, desc="Generating Slices"):
            if widget.angles_checkbox.isChecked():
                sino, angles = load_angles_and_create_sinograms(viewer, experiment, projs, cor)
            else:
                sino = create_sinogram_slice(projs, 2 * cor, slice_idx)
                angles = get_angles(viewer, experiment, 2 * sino.shape[0], full=False) if widget.angles_checkbox.isChecked() else create_angles(sino, end=np.pi)
            slice_ = apply_mask_and_reconstruct(sino, angles, sigma, coeff, apply_unsharp=apply_unsharp)
            if target_shape is None:
                target_shape = slice_.shape
            slices.append(resize_to_target(slice_, target_shape))

        result = {'slice': np.array(slices)}
        add_image_to_layer(result, "cor_test", viewer)
        clear_memory([projs])
    except Exception as e:
        print(f"Error during half COR test: {e}")
    finally:
        dialog.close()

def call_process_one_slice(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        # Mise à jour des paramètres de base
        base_parameters = ["slice_idx", "center_of_rotation", "acquisition_type", "double_flatfield"]
        experiment.update_parameters(widget, parameters_to_update=base_parameters)

        slice_idx = int(experiment.slice_idx)
        sigma, coeff = (float(experiment.sigma), float(experiment.coeff)) if widget.paganin_checkbox.isChecked() else (0, 0)
        cor = int(experiment.center_of_rotation)

        # Vérification si Paganin doit être appliqué
        if widget.paganin_checkbox.isChecked():
            projs = get_projections(viewer, 'paganin', fallback_func=lambda: call_paganin(experiment, viewer, widget, one_slice=True))['paganin']
            apply_unsharp = True
        else:
            projs = get_projections(viewer, 'preprocess', fallback_func=lambda: call_preprocess(experiment, viewer, widget))['preprocess']
            apply_unsharp = False

        if projs.ndim != 2:
            projs = projs[:, slice_idx]

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']

        
        projs = cp.asarray(projs)

        if widget.acquisition_type_selection.currentIndex() == 0:
            sinogram = pad_and_shift_projection(projs, cor)
            angles = create_angles(sinogram, end=2 * np.pi)
        else:
            if widget.angles_checkbox.isChecked():
                sinogram, angles = load_angles_and_create_sinograms(viewer, experiment, projs, cor)
            else:
                sinogram = create_sinogram_slice(projs, 2 * cor)
                angles = create_angles(sinogram, end=np.pi)

        if not experiment.bigdata:
            add_image_to_layer({'sinogram': sinogram}, f"cor_{cor}", viewer)

        slice_ = apply_mask_and_reconstruct(sinogram, angles, sigma, coeff, apply_unsharp=apply_unsharp)
        result = {'slice': np.array(slice_)}

        add_image_to_layer(result, f"cor_{cor}", viewer)
        clear_memory([projs])
        return result
    except Exception as e:
        print(f"Error during single slice processing: {e}")
    finally:
        dialog.close()

def call_process_all_slices(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        # Mise à jour des paramètres de base
        base_parameters = ["center_of_rotation", "acquisition_type", "double_flatfield"]
        experiment.update_parameters(widget, parameters_to_update=base_parameters)

        # Vérification si Paganin doit être appliqué
        if widget.paganin_checkbox.isChecked():
            projs = get_projections(viewer, 'paganin', fallback_func=lambda: call_paganin(experiment, viewer, widget))['paganin']
            apply_unsharp = True
        else:
            projs = get_projections(viewer, 'preprocess', fallback_func=lambda: call_preprocess(experiment, viewer, widget))['preprocess']
            apply_unsharp = False

        cor = int(experiment.center_of_rotation)
        sigma, coeff = (float(experiment.sigma), float(experiment.coeff)) if widget.paganin_checkbox.isChecked() else (0, 0)

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)['double_flatfield_corrected']

        n_slices, width = projs.shape[1], projs.shape[-1]
        recon = np.zeros((n_slices, width, width), dtype=np.float32) if widget.acquisition_type_selection.currentIndex() == 0 else None

        if widget.acquisition_type_selection.currentIndex() == 0:
            for i in tqdm(range(n_slices), desc="Processing all slices"):
                sino = pad_and_shift_projection(cp.asarray(projs[:, i]), cor)
                angles = get_angles(viewer, experiment, sino.shape[0]) if widget.angles_checkbox.isChecked() else create_angles(sino, end=2*np.pi)
                slice_ = apply_mask_and_reconstruct(sino, angles, sigma, coeff, apply_unsharp=apply_unsharp)
                recon[i] = slice_
        else:
            if widget.angles_checkbox.isChecked():
                sino, angles = load_angles_and_create_sinograms(viewer, experiment, projs, cor)

            else:
                sino = create_sinogram(cp.asarray(projs), 2 * cor)
                angles = get_angles(viewer, experiment, 2 * sino.shape[0], full=False) if widget.angles_checkbox.isChecked() else create_angles(sino, end=np.pi)
            recon = np.zeros((sino.shape[0], sino.shape[2], sino.shape[2]), dtype=np.float32)
            for i in tqdm(range(sino.shape[0]), desc="Processing all slices"):
                slice_ = apply_mask_and_reconstruct(sino[i], angles, sigma, coeff, apply_unsharp=apply_unsharp)
                recon[i] = slice_

        result = {'Vol': recon}
        add_image_to_layer(result, f"c{cor}_db{experiment.db}_s{sigma}_c{coeff}", viewer)
        clear_memory([projs])
        return result
    except Exception as e:
        print(f"Error during full volume processing: {e}")
    finally:
        dialog.close()