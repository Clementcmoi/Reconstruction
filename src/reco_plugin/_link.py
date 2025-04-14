# Imports
from cupyx.scipy.ndimage import shift
import numpy as np
import cupy as cp
from tqdm import tqdm
from skimage.transform import resize
import gc

# Local imports
from .processing.cor import *
from .processing.phase import paganin_filter, unsharp_mask, paganin_filter_slice
from .processing.process import apply_flat_darkfield, double_flatfield_correction
from .processing.reconstruction import reconstruct_from_sinogram_slice, create_angles, create_disk_mask
from .processing.sinogram import create_sinogram, create_sinogram_slice
from .utils.qt_helpers import create_processing_dialog, PlotWindow


# Utility functions
def add_image_to_layer(results, img_name, viewer):
    """Add processed images to the viewer."""
    for name, image in results.items():
        viewer.add_image(image.real, name=f"{name}_{img_name}")


def clear_memory(variables):
    """
    Clear variables from RAM and GPU memory.
    """
    for var in variables:
        del var
    gc.collect()  # Clear RAM
    cp._default_memory_pool.free_all_blocks()  # Clear GPU memory


# Preprocessing
def call_preprocess(experiment, viewer, widget):
    """Apply flat and dark field correction to the sample images."""
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=["sample_images", "darkfield", "flatfield", "bigdata"])
        print("Preprocessing...")

        sample = np.transpose(viewer.layers[experiment.sample_images].data, viewer.dims.order)
        dark = np.mean(viewer.layers[experiment.darkfield].data, axis=0) if experiment.darkfield else None
        flat = np.mean(viewer.layers[experiment.flatfield].data, axis=0) if experiment.flatfield else None

        corrected = apply_flat_darkfield(sample, flat, dark)

        # Add to viewer only if bigdata is False
        if not experiment.bigdata:
            add_image_to_layer(corrected, experiment.sample_images, viewer)

        # Clear memory
        clear_memory([sample, dark, flat])

        return corrected
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    finally:
        dialog.close()


# Paganin filter
def call_paganin(experiment, viewer, widget, one_slice=False):
    """Apply Paganin phase retrieval filter."""
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "pixel", "effective_pixel", "dist_object_detector", "energy", "db", "sigma", "coeff"
        ])

        # Retrieve projections
        projs = next((layer.data for layer in viewer.layers if layer.name.startswith('preprocess')), None)
        if projs is None:
            projs = call_preprocess(experiment, viewer, widget)['preprocess']

        # Apply Paganin filter
        if widget.paganin_checkbox.isChecked():
            if one_slice:
                result = paganin_filter_slice(
                    projs, int(experiment.slice_idx), float(experiment.energy),
                    float(experiment.pixel), float(experiment.effective_pixel),
                    float(experiment.dist_object_detector), float(experiment.db)
                )
            else:
                result = paganin_filter(
                    projs, float(experiment.energy), float(experiment.pixel),
                    float(experiment.effective_pixel), float(experiment.dist_object_detector),
                    float(experiment.db)
                )

            # Add to viewer only if bigdata is False
            if not experiment.bigdata:
                add_image_to_layer(result, experiment.sample_images, viewer)

            # Clear memory
            clear_memory([projs])

            return result
        else:
            print("Paganin checkbox is not checked.")
            return projs
    except Exception as e:
        print(f"Error during Paganin: {e}")
    finally:
        dialog.close()


# Standard COR test
def call_standard_cor_test(experiment, viewer, widget):
    """Perform standard center-of-rotation (COR) test."""
    processing_dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "cor_min", "cor_max", "cor_step", "double_flatfield"
        ])

        slice_idx = int(experiment.slice_idx)
        sigma = float(experiment.sigma)
        coeff = float(experiment.coeff)
        cor_min, cor_max, cor_step = map(int, [experiment.cor_min, experiment.cor_max, experiment.cor_step])

        # Retrieve projections
        projs = next((layer.data[:, slice_idx] if layer.ndim == 3 else layer.data
                      for layer in viewer.layers if layer.name.startswith('paganin')), None)
        if projs is None:
            projs = call_paganin(experiment, viewer, widget, one_slice=True)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        # Generate slices for COR candidates
        cor_candidate = np.arange(cor_min, cor_max + cor_step, cor_step)
        slices, target_shape = [], None
        projs = cp.asarray(projs)

        for cor in tqdm(cor_candidate, desc="Generating Slices"):
            pad_width = abs(cor)
            padded_projs = cp.pad(projs, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
            sinogram_shifted = shift(padded_projs, (0, cor), order=1, mode='constant').get()

            angles = create_angles(sinogram_shifted, end=2 * np.pi)
            disk = create_disk_mask(sinogram_shifted)
            slice_ = reconstruct_from_sinogram_slice(sinogram_shifted, angles) * disk
            slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()

            # Resize to match target shape
            if target_shape is None:
                target_shape = slice_.shape
            else:
                slice_ = resize(slice_, target_shape, mode='constant', anti_aliasing=True)

            slices.append(slice_)

        slices = {'slice': np.array(slices)}

        # Add to viewer only if bigdata is False
        if not experiment.bigdata:
            add_image_to_layer(slices, f"cor_test", viewer)

        # Clear memory
        clear_memory([projs, padded_projs, sinogram_shifted])

        return slices
    except Exception as e:
        print(f"Error during Standard COR test: {e}")
    finally:
        processing_dialog.close()


# Global COR calculation
def call_find_global_cor(experiment, viewer, widget):
    """Calculate global center-of-rotation (COR)."""
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])

        # Retrieve projections
        projs = next((layer.data for layer in viewer.layers if layer.name.startswith('paganin')), None)
        if projs is None:
            projs = call_paganin(experiment, viewer, widget)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        cor, plot_data = calc_cor(projs)

        cor = cor[np.isfinite(cor)]
        cor_std = np.std(cor, axis=0)
        cor_mean = np.mean(cor)
        mask_cor = (cor > cor_mean - cor_std) & (cor < cor_mean + cor_std)
        cor_mean = np.mean(cor[mask_cor])
        widget.center_of_rotation_input.setText(str(cor_mean))

        # Ensure cor is passed as an array to the PlotWindow
        widget.plot_window = PlotWindow(plot_data, cor_values=cor)
        widget.plot_window.show()

    except Exception as e:
        print(f"Error during global COR calculation: {e}")
    finally:
        dialog.close()


# Half COR test
def call_half_cor_test(experiment, viewer, widget):
    """Perform half center-of-rotation (COR) test."""
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "double_flatfield", "center_of_rotation", "cor_fenetre"
        ])

        slice_idx = int(experiment.slice_idx)
        sigma = float(experiment.sigma)
        coeff = float(experiment.coeff)
        cor_test = int(experiment.center_of_rotation)
        cor_fenetre = int(experiment.cor_fenetre)

        # Retrieve projections
        projs = next((layer.data[:, slice_idx] for layer in viewer.layers if layer.name.startswith('paganin')), None)
        if projs is None:
            projs = call_paganin(experiment, viewer, widget, one_slice=True)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        cor_candidate = np.arange(cor_test - cor_fenetre, cor_test + cor_fenetre, 1)
        projs = cp.asarray(projs)

        slices, target_shape = [], None

        for cor in tqdm(cor_candidate, desc="Generating Slices"):
            sinogram = create_sinogram_slice(projs, 2 * cor, slice_idx).get()

            angles = create_angles(sinogram, end=np.pi)
            disk = create_disk_mask(sinogram)
            slice_ = reconstruct_from_sinogram_slice(sinogram, angles) * disk
            slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()

            # Resize to match target shape
            if target_shape is None:
                target_shape = slice_.shape
            else:
                slice_ = resize(slice_, target_shape, mode='constant', anti_aliasing=True)

            slices.append(slice_)

        slices = {'slice': np.array(slices)}
        add_image_to_layer(slices, f"cor_test", viewer)

        # Clear memory
        clear_memory([projs, sinogram])

    except Exception as e:
        print(f"Error during global COR calculation: {e}")
    finally:
        dialog.close()


# Process one slice
def call_process_one_slice(experiment, viewer, widget):
    """Process a single slice."""
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])

        slice_idx = int(experiment.slice_idx)
        sigma = float(experiment.sigma)
        coeff = float(experiment.coeff)
        cor = int(experiment.center_of_rotation)

        # Retrieve projections
        projs = next((layer.data[:, slice_idx] if layer.ndim == 3 else layer.data
                      for layer in viewer.layers if layer.name.startswith('paganin')), None)
        if projs is None:
            projs = call_paganin(experiment, viewer, widget, one_slice=True)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        projs = cp.asarray(projs)

        if widget.acquisition_type_selection.currentIndex() == 0:
            # Pad the image to accommodate the shift
            pad_width = abs(cor)
            padded_projs = cp.pad(projs, ((0, 0), (pad_width, pad_width)), mode='constant', constant_values=0)
            sinogram = shift(padded_projs, (0, cor), order=1, mode='constant').get()
            angles = create_angles(sinogram, end=2 * np.pi)
        else:
            sinogram = create_sinogram_slice(projs, 2 * cor, slice_idx).get()
            angles = create_angles(sinogram, end=np.pi)

        if not experiment.bigdata:
            add_image_to_layer({'sinogram': sinogram}, f"cor_{cor}", viewer)

        mask = create_disk_mask(sinogram)
        slice_ = reconstruct_from_sinogram_slice(sinogram, angles) * mask
        slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()

        result = {'slice': np.array(slice_)}
        add_image_to_layer(result, f"cor_{cor}", viewer)

        # Clear memory
        clear_memory([projs, padded_projs, sinogram])

        return result

    except Exception as e:
        print(f"Error during slice reconstruction test: {e}")
    finally:
        dialog.close()


# Process all slices
def call_process_all_slices(experiment, viewer, widget):
    """Process all slices."""
    dialog = create_processing_dialog(viewer.window._qt_window)
    try:
        experiment.update_parameters(widget, parameters_to_update=[
            "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])

        sigma = float(experiment.sigma)
        coeff = float(experiment.coeff)
        cor = int(experiment.center_of_rotation)

        # Retrieve projections
        projs = next((layer.data for layer in viewer.layers if layer.name.startswith('paganin')), None)
        if projs is None:
            projs = call_paganin(experiment, viewer, widget)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        n_slices = projs.shape[1]
        width = projs.shape[-1]

        if widget.acquisition_type_selection.currentIndex() == 0:
            reconstruction = np.zeros((n_slices, width, width), dtype=np.float32)
            for i in tqdm(range(n_slices), desc="Generating Slices"):
                sinogram = cp.asarray(projs[:, i])
                sinogram = shift(sinogram, (0, cor), order=1, mode='constant')
                angles = create_angles(sinogram, end=2 * np.pi)
                mask = create_disk_mask(sinogram)
                slice_ = reconstruct_from_sinogram_slice(sinogram.get(), angles) * mask
                slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()
                reconstruction[i] = slice_
        else:
            sinogram = create_sinogram(projs, 2 * cor)
            angles = create_angles(sinogram, end=np.pi)
            reconstruction = np.zeros((sinogram.shape[0], sinogram.shape[2], sinogram.shape[2]), dtype=np.float32)
            mask = create_disk_mask(sinogram)
            for i in tqdm(range(sinogram.shape[0]), desc="Generating Slices"):
                slice_ = reconstruct_from_sinogram_slice(sinogram[i], angles) * mask
                slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()
                reconstruction[i] = slice_

        # Create a detailed description of the processing steps
        desc = f"c{cor}_db{experiment.db}_s{sigma}_c{coeff}"

        result = {'Vol': np.array(reconstruction)}
        add_image_to_layer(result, desc, viewer)

        # Clear memory
        clear_memory([projs, sinogram])

        return result

    except Exception as e:
        print(f"Error during all-slice reconstruction: {e}")
    finally:
        dialog.close()