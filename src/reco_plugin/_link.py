from cupyx.scipy.ndimage import shift
import numpy as np
import cupy as cp
from tqdm import tqdm
from skimage.transform import resize  # Add this import for resizing

from .processing.cor import *
from .processing.phase import paganin_filter, unsharp_mask
from .processing.process import apply_flat_darkfield, double_flatfield_correction
from .processing.reconstruction import reconstruct_from_sinogram_slice, create_angles, create_disk_mask
from .processing.sinogram import create_sinogram, create_sinogram_slice

from .utils.qt_helpers import create_processing_dialog, PlotWindow


def add_image_to_layer(results, img_name, viewer):
    for name, image in results.items():
        viewer.add_image(image.real, name=f"{name}_{img_name}")


def call_preprocess(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window.qt_viewer)
    try:
        print("Preprocessing...")
        # Save only relevant parameters for preprocessing
        experiment.update_parameters(widget, parameters_to_update=["sample_images", "darkfield", "flatfield"])
        experiment.save_settings(parameters_to_save=["sample_images", "darkfield", "flatfield"])

        sample = np.transpose(viewer.layers[experiment.sample_images].data, viewer.dims.order)

        dark = flat = None
        if experiment.darkfield:
            dark = np.mean(viewer.layers[experiment.darkfield].data, axis=0)
        if experiment.flatfield:
            flat = np.mean(viewer.layers[experiment.flatfield].data, axis=0)

        corrected = apply_flat_darkfield(sample, flat, dark)
        add_image_to_layer(corrected, experiment.sample_images, viewer)
        return corrected

    except Exception as e:
        print(f"Error during preprocessing: {e}")
    finally:
        dialog.close()


def call_paganin(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window.qt_viewer)
    try:
        # Save only relevant parameters for Paganin
        experiment.update_parameters(widget, parameters_to_update=[
            "pixel", "effective_pixel", "dist_object_detector", "energy", "db", "sigma", "coeff"
        ])
        experiment.save_settings(parameters_to_save=[
            "pixel", "effective_pixel", "dist_object_detector", "energy", "db", "sigma", "coeff"
        ])

        for layer in viewer.layers:
            if layer.name.startswith('preprocess'):
                projs = layer.data
                break
        else:
            projs = call_preprocess(experiment, viewer, widget)['preprocess']

        if widget.paganin_checkbox.isChecked():
            result = paganin_filter(
                projs,
                float(experiment.energy),
                float(experiment.pixel),
                float(experiment.effective_pixel),
                float(experiment.dist_object_detector),
                float(experiment.db)
            )
            add_image_to_layer(result, experiment.sample_images, viewer)
            return result
        else:
            print("Paganin checkbox is not checked.")
            return projs

    except Exception as e:
        print(f"Error during Paganin: {e}")
    finally:
        dialog.close()

def call_standard_cor_test(experiment, viewer, widget):
    processing_dialog = create_processing_dialog(viewer.window.qt_viewer)

    try:
        # Save only relevant parameters for the Standard COR test
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "cor_min", "cor_max", "cor_step", "double_flatfield"
        ])
        experiment.save_settings(parameters_to_save=[
            "slice_idx", "sigma", "coeff", "cor_min", "cor_max", "cor_step", "double_flatfield"
        ])

        slice_idx = int(widget.slice_selection.value())
        sigma = float(widget.sigma_input.text())
        coeff = float(widget.coeff_input.text())
        cor_min = int(widget.cor_min_input.text())
        cor_max = int(widget.cor_max_input.text())
        cor_step = int(widget.cor_step_input.text())

        for layer in viewer.layers:
            if layer.name.startswith('paganin'):
                projs = layer.data
                break
        else:
            projs = call_paganin(experiment, viewer, widget)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        cor_candidate = np.arange(cor_min, cor_max + cor_step, cor_step)

        slices = []
        target_shape = None  # Initialize target shape

        for cor in tqdm(cor_candidate, desc="Generating Slices"):
            sinogram = cp.asarray(projs[:, slice_idx])
            sinogram = shift(sinogram, (0, cor), order=1, mode='constant')

            angles = create_angles(sinogram, end=2 * np.pi)
            disk = create_disk_mask(sinogram)
            slice_ = reconstruct_from_sinogram_slice(sinogram.get(), angles) * disk

            slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()

            if target_shape is None:
                target_shape = slice_.shape  # Set target shape based on the first slice
            else:
                slice_ = resize(slice_, target_shape, mode='constant', anti_aliasing=True)  # Resize to match target shape

            slices.append(slice_)

        slices = {'slice': np.array(slices)}
        add_image_to_layer(slices, f"cor_test", viewer)

        return slices
    except Exception as e:
        print(f"Error during Standard COR test: {e}")
    finally:
        processing_dialog.close()


def call_find_global_cor(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window.qt_viewer)
    try:
        # Save only relevant parameters for processing one slice
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])
        experiment.save_settings(parameters_to_save=[
            "slice_idx", "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])

        for layer in viewer.layers:
            if layer.name.startswith('paganin'):
                projs = layer.data
                break
        else:
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

def call_half_cor_test(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window.qt_viewer)

    try:
        # Save only relevant parameters for the Standard COR test
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "double_flatfield"
        ])
        experiment.save_settings(parameters_to_save=[
            "slice_idx", "sigma", "coeff", "double_flatfield"
        ])

        slice_idx = int(widget.slice_selection.value())
        sigma = float(widget.sigma_input.text())
        coeff = float(widget.coeff_input.text())
        cor_test = int(widget.center_of_rotation_input.text())
        cor_fentre = int(widget.fenetre_input.value())

        for layer in viewer.layers:
            if layer.name.startswith('paganin'):
                projs = layer.data
                break
        else:
            projs = call_paganin(experiment, viewer, widget)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        cor_candidate = np.arange(cor_test - cor_fentre, cor_test + cor_fentre, 1)
        projs = cp.asarray(projs[:, slice_idx])

        slices = []
        target_shape = None  # Initialize target shape

        for cor in tqdm(cor_candidate, desc="Generating Slices"):
            sinogram = create_sinogram_slice(projs, 2 * cor, slice_idx).get()

            angles = create_angles(sinogram, end=np.pi)
            disk = create_disk_mask(sinogram)
            slice_ = reconstruct_from_sinogram_slice(sinogram, angles) * disk

            slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()

            if target_shape is None:
                target_shape = slice_.shape  # Set target shape based on the first slice
            else:
                slice_ = resize(slice_, target_shape, mode='constant', anti_aliasing=True)  # Resize to match target shape

            slices.append(slice_)

        slices = {'slice': np.array(slices)}
        add_image_to_layer(slices, f"cor_test", viewer)

    except Exception as e:
        print(f"Error during global COR calculation: {e}")
    finally:
        dialog.close()

def call_process_one_slice(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window.qt_viewer)
    try:
        # Save only relevant parameters for processing one slice
        experiment.update_parameters(widget, parameters_to_update=[
            "slice_idx", "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])
        experiment.save_settings(parameters_to_save=[
            "slice_idx", "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])

        slice_idx = int(widget.slice_selection.value())
        sigma = float(widget.sigma_input.text())
        coeff = float(widget.coeff_input.text())
        cor = int(widget.center_of_rotation_input.text())

        for layer in viewer.layers:
            if layer.name.startswith('paganin'):
                projs = layer.data
                break
        else:
            projs = call_paganin(experiment, viewer, widget)['paganin']

        if widget.double_flatfield_checkbox.isChecked():
            projs = double_flatfield_correction(projs)

        if widget.acquisition_type_selection.currentIndex() == 0:
            sinogram = cp.asarray(projs[:, slice_idx])
            sinogram = shift(sinogram, (0, cor), order=1, mode='constant').get()
            angles = create_angles(sinogram, end=2 * np.pi)
        else:
            sinogram = create_sinogram_slice(projs, 2 * cor, slice_idx)
            angles = create_angles(sinogram, end=np.pi)

        add_image_to_layer({'sinogram': sinogram}, f"cor_{cor}", viewer)

        mask = create_disk_mask(sinogram)
        slice_ = reconstruct_from_sinogram_slice(sinogram, angles) * mask
        slice_ = unsharp_mask(cp.asarray(slice_), sigma=sigma, coeff=coeff).get()

        result = {'slice': np.array(slice_)}
        add_image_to_layer(result, f"cor_{cor}", viewer)
        return result

    except Exception as e:
        print(f"Error during slice reconstruction test: {e}")
    finally:
        dialog.close()


def call_process_all_slices(experiment, viewer, widget):
    dialog = create_processing_dialog(viewer.window.qt_viewer)
    try:
        # Save only relevant parameters for processing all slices
        experiment.update_parameters(widget, parameters_to_update=[
            "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])
        experiment.save_settings(parameters_to_save=[
            "sigma", "coeff", "center_of_rotation", "acquisition_type", "double_flatfield"
        ])

        sigma = float(widget.sigma_input.text())
        coeff = float(widget.coeff_input.text())
        cor = int(widget.center_of_rotation_input.text())

        for layer in viewer.layers:
            if layer.name.startswith('paganin'):
                projs = layer.data
                break
        else:
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

        result = {'slice': np.array(reconstruction)}
        add_image_to_layer(result, f"cor_{cor}", viewer)
        return result

    except Exception as e:
        print(f"Error during all-slice reconstruction: {e}")
    finally:
        dialog.close()