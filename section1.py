from qtpy.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
import math
from cupy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from cupyx.scipy.ndimage import gaussian_filter
from numpy import pi
import numpy as np
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.draw import disk
import astra
import cupy as cp

def keVtoLambda(energy_kev):
    """
    Convert energy in keV to wavelength in m.

    Parameters
    ----------
    energy_kev : float
        Energy in keV

    Returns
    -------
    float
        Wavelength in m
    """
    h = 6.58211928e-19  # Planck constant in keV·s
    c = 299792458       # Speed of light in m/s
    return h * c / energy_kev

def phase_retrieval(I, delta_beta, dist_obj_detector, energy_kev, pixel_size):
    """
    Apply phase retrieval according to formula (6) from Weitkamp et al. (2011).

    Parameters
    ----------
    I : cupy.ndarray
        Measured intensity image.
    delta_beta : float
        Beta/delta ratio.
    dist_obj_detector : float
        Propagation distance (m).
    energy_kev : float
        X-ray energy (keV).
    pixel_size : float
        Detector pixel size (m).

    Returns
    -------
    phi : numpy.ndarray
        Reconstructed phase map.
    """
    wavelength = keVtoLambda(energy_kev)
    ny, nx = I.shape

    # Spatial frequencies
    u = fftfreq(nx, d=pixel_size)
    v = fftfreq(ny, d=pixel_size)
    U, V = cp.meshgrid(u, v, indexing='ij')

    # Fourier transform of the image
    I_fft = fftshift(fft2(I))

    # Compute filter
    denominator = 1 + cp.pi * wavelength * dist_obj_detector * delta_beta * (U**2 + V**2)
    denominator[denominator == 0] = cp.finfo(float).eps  # Avoid division by zero

    # Apply filter and inverse FFT
    phi_fft = I_fft / denominator
    phi = - cp.log(cp.maximum(cp.real(ifft2(ifftshift(phi_fft))), cp.finfo(cp.float32).eps)) * delta_beta * 0.5

    return phi

def get_padding_size(image, energy, effective_pixel_size, distance):
    """
    Calculate the padding size for a 2D image.

    Parameters
    ----------
    image : cupy.ndarray
        2D array of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).

    Returns
    -------
    tuple
        Padded sizes (nx_padded, ny_padded)
    """
    ny, nx = image.shape
    wavelength = keVtoLambda(energy)

    n_margin = math.ceil(3 * wavelength * distance / (2 * effective_pixel_size ** 2))
    nx_margin = nx + 2 * n_margin
    ny_margin = ny + 2 * n_margin

    nx_padded = int(2 ** math.ceil(math.log2(nx_margin)))
    ny_padded = int(2 ** math.ceil(math.log2(ny_margin)))

    return nx_padded, ny_padded

def padding(image, energy, effective_pixel_size, distance):
    """
    Pad a 2D image to avoid edge artifacts during phase retrieval with the closest value.

    Parameters
    ----------
    image : cupy.ndarray
        2D array of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).

    Returns
    -------
    tuple
        Padded image, padded nx, padded ny
    """
    ny, nx = image.shape
    nx_padded, ny_padded = get_padding_size(image, energy, effective_pixel_size, distance)

    top = (ny_padded - ny) // 2
    bottom = ny_padded - ny - top
    left = (nx_padded - nx) // 2
    right = nx_padded - nx - left

    padded_image = cp.pad(image, ((top, bottom), (left, right)), mode='reflect')
    return padded_image, nx_padded, ny_padded

def unsharp_mask(image: cp.ndarray, sigma: float = 1.0, coeff: float = 1.0) -> cp.ndarray:
    blurred = gaussian_filter(image, sigma=sigma, mode='reflect')
    return (1 + coeff) * image - coeff * blurred

def process_projection(proj, nx, ny, energy, effective_pixel_size, distance, delta_beta, pixel_size, sigma, coeff):
    """
    Process a single projection image.

    Parameters
    ----------
    proj : cupy.ndarray
        Projection image.
    nx : int
        Original width of the image.
    ny : int
        Original height of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).
    delta_beta : float
        Ratio beta/delta for phase retrieval.
    pixel_size : float
        Detector pixel size (m).

    Returns
    -------
    numpy.ndarray
        Cropped phase-retrieved projection.
    """

    proj_cp = cp.asarray(proj)
    padded_proj, nx_padded, ny_padded = padding(proj_cp, energy, effective_pixel_size, distance)
    retrieved_proj = phase_retrieval(padded_proj, delta_beta, distance, energy, pixel_size)
    retrieved_proj = unsharp_mask(retrieved_proj, sigma=sigma, coeff=coeff)

    x_margin = (nx_padded - nx) // 2
    y_margin = (ny_padded - ny) // 2

    return retrieved_proj[y_margin:y_margin + ny, x_margin:x_margin + nx].get()

def paganin_filter(projs, pixel_size, effective_pixel_size, distance, energy, delta_beta, sigma, coeff):
    """
    Apply Paganin filter to a set of projections.

    Parameters
    ----------
    projs : cupy.ndarray
        3D array of projection images.
    pixel_size : float
        Detector pixel size (m).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).
    energy : float
        Energy of the X-ray beam (keV).
    delta_beta : float
        Ratio beta/delta for phase retrieval.
    sigma : float
        Standard deviation for Gaussian filter.
    coeff : float
        Coefficient for unsharp mask.
    
    Returns
    -------
    projs : cupy.ndarray
        3D array of phase-retrieved projections.
    """

    retrieved_projs = np.zeros(projs.shape, dtype=np.float32)
    for i in tqdm(range(projs.shape[0]), desc='Processing projections'):
        retrieved_projs[i] = process_projection(projs[i], projs.shape[2], projs.shape[1], energy, effective_pixel_size, distance, delta_beta, pixel_size, sigma, coeff)

    return retrieved_projs


def double_flatfield_correction(projs):
    """
    Apply double flat-field correction to an image.

    Parameters
    ----------
    proj : 2D numpy array
        Image to correct

    Returns
    -------
    I_corr : 2D numpy array
        Corrected image

    """
    mean_proj = np.mean(projs, axis=0)

    mean_proj[mean_proj == 0] = 1e-6

    I_corr = projs / mean_proj

    return I_corr

def apply_left_weighting(projs, CoR):
    """
    Applique un poids linéaire sur la partie gauche des projections.
    """
    weights = np.linspace(0, 1, CoR)[None, None, :]
    projs[:, :, :CoR] *= weights

    return projs 
   
def create_sinogram_slice(projs, CoR, slice_idx):
    """
    Create a sinogram from a set of projections.
    """
    theta, _, ny = projs.shape

    sino = np.zeros((theta//2, 2 * ny - CoR))

    flip = projs[:theta // 2, slice_idx, ::-1]  # np.flip optimisé

    sino[:, :ny] += flip
    sino[:,  -ny:] += projs[theta//2:, slice_idx, :]

    return sino

def create_sinogram(projs, CoR):
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

def from_degress_to_radians(angles):
    return angles * pi / 180

def from_radians_to_degrees(angles):
    return angles * 180 / pi

def create_angles(sinogram):
    angles = np.linspace(0, pi, sinogram.shape[1], endpoint=False)
    print(f"Angles: {angles}")
    return angles

def reconstruct_from_sinogram_slice(sinogram, angles):
    """
    Reconstruct a 2D image from a sinogram using FBP_CUDA algorithm from ASTRA Toolbox.

    Parameters:
    - sinogram: 2D numpy array (angles, detectors) containing the sinogram.
    - angles: 1D numpy array of rotation angles (in radians).

    Returns:
    - reconstruction: 2D numpy array representing the reconstructed image.
    """

    # Définition des géométries de projection et du volume
    proj_geom = astra.create_proj_geom('parallel', 1.0, sinogram.shape[1], angles)
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

    return reconstruction

def create_disk_mask(sinogram):
    """
    Create a circular disk mask for the sinogram.
    """
    disk_mask = np.zeros((sinogram.shape[2], sinogram.shape[2]))
    rr, cc = disk((sinogram.shape[2] // 2, sinogram.shape[2] // 2), sinogram.shape[2] // 2)
    disk_mask[rr, cc] = 1

    return disk_mask

def create_processing_dialog(parent, message="Processing..."):
    """
    Create and display a dialog with a message to indicate that processing is ongoing.
    """
    dialog = QDialog(parent)
    dialog.setWindowTitle("Processing")
    layout = QVBoxLayout()
    label = QLabel(message)
    layout.addWidget(label)
    dialog.setLayout(layout)
    dialog.setFixedSize(200, 100)
    dialog.show()
    QApplication.processEvents()
    return dialog

def apply_corrections(viewer, experiment):
    """
    Apply flatfield and darkfield corrections to the sample layers 
    using the data stored in the experiment object.
    """
    print("Applying corrections")
    sample_layer = viewer.layers[experiment.sample_images].data

    # Adjust for axis order changes in Napari
    sample_layer = np.transpose(sample_layer, viewer.dims.order)

    if experiment.darkfield is not None:
        darkfield_layer = np.median(viewer.layers[experiment.darkfield].data)
        sample_layer = sample_layer - darkfield_layer

    if experiment.flatfield is not None:
        flatfield_layer = np.median(viewer.layers[experiment.flatfield].data)
        sample_layer = sample_layer / flatfield_layer

    return sample_layer

def apply_corrections_one_slice(viewer, experiment):
    """
    Apply flatfield and darkfield corrections to the sample layers 
    using the data stored in the experiment object.
    """
    print("Applying corrections")
    slice_idx = experiment.slice_idx

    sample_layer = viewer.layers[experiment.sample_images].data

    # Check if the data is 2D or 3D
    if sample_layer.ndim == 3:
        # Adjust for axis order changes in Napari
        sample_layer = np.transpose(sample_layer, viewer.dims.order)
        sample_slice = sample_layer[slice_idx]
    else:
        sample_slice = sample_layer

    if experiment.darkfield is not None:
        darkfield_layer = np.mean(viewer.layers[experiment.darkfield].data) if sample_layer.ndim == 3 else viewer.layers[experiment.darkfield].data
        sample_slice = sample_slice - darkfield_layer

    if experiment.flatfield is not None:
        flatfield_layer = np.mean(viewer.layers[experiment.flatfield].data) if sample_layer.ndim == 3 else viewer.layers[experiment.flatfield].data
        sample_slice = sample_slice / flatfield_layer

    return sample_slice

def add_image_to_layer(results, method, viewer):
    """
    Add the resulting image to the viewer as a new layer.
    """
    for name, image in results.items():
        viewer.add_image(image.real, name=f"{name}_{method}", )

def process_try_paganin(experiment, viewer):

    print(viewer.dims.order)

    processing_dialog = create_processing_dialog(viewer.window.qt_viewer)

    try:
        sample_layer = apply_corrections(viewer, experiment)

        energy = experiment.energy
        pixel_size = experiment.pixel
        effective_pixel_size = experiment.effective_pixel
        db = experiment.db
        dist_object_detector = experiment.dist_object_detector
        sigma = experiment.sigma
        coeff = experiment.coeff

        projs = paganin_filter(sample_layer, 
                    pixel_size, effective_pixel_size, dist_object_detector, energy, db, sigma, coeff)
        
        add_image_to_layer({"Reconstruction": projs}, "FBP", viewer)

    except Exception as e:
        print(f"Error processing slice: {e}")

    finally:
        processing_dialog.close()


def process_all_slices(experiment, viewer):
    processing_dialog = create_processing_dialog(viewer.window.qt_viewer)

    try:
        sample_layer = apply_corrections(viewer, experiment)

        energy = experiment.energy
        pixel_size = experiment.pixel
        effective_pixel_size = experiment.effective_pixel
        db = experiment.db
        dist_object_detector = experiment.dist_object_detector
        sigma = experiment.sigma
        coeff = experiment.coeff

        projs = paganin_filter(sample_layer, 
                    pixel_size, effective_pixel_size, dist_object_detector, energy, db, sigma, coeff)

        if experiment.double_flatfield:
            print("Applying double flatfield correction")
            projs = double_flatfield_correction(projs)
            add_image_to_layer({"Double Flatfield Correction": projs}, "DFC", viewer)

        if experiment.center_of_rotation is not None:
            print("Creating sinogram from half acquisition with center of rotation : ", experiment.center_of_rotation)
            CoR = round(2 * experiment.center_of_rotation)
            sinogram = create_sinogram(projs, CoR)
        else:
            print("Creating sinogram from full acquisition")
            sinogram = np.swapaxes(projs, 0, 1)

        angles = create_angles(sinogram)
        disk_mask = create_disk_mask(sinogram)

        reconstruction = np.zeros((sinogram.shape[0], sinogram.shape[2], sinogram.shape[2]))

        print("Reconstructing slices")
        for i in tqdm(range(sinogram.shape[0])):
            reconstruction[i] = reconstruct_from_sinogram_slice(sinogram[i], angles) * disk_mask

        add_image_to_layer({"Reconstruction": reconstruction}, "FBP", viewer)

    except Exception as e:
        print(f"Error processing all slices: {e}")

    finally:
        processing_dialog.close()




def unsharp_mask(image: cp.ndarray, sigma: float = 1.0, coeff: float = 1.0) -> cp.ndarray:
    blurred = gaussian_filter(image, sigma=sigma, mode='reflect')
    return (1 + coeff) * image - coeff * blurred

import math
from cupy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from cupyx.scipy.ndimage import gaussian_filter, median_filter
from numpy import pi
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.draw import disk
import astra
import cupy as cp

import matplotlib.pyplot as plt

def keVtoLambda(energy_kev):
    """
    Convert energy in keV to wavelength in m.

    Parameters
    ----------
    energy_kev : float
        Energy in keV

    Returns
    -------
    float
        Wavelength in m
    """
    h = 6.58211928e-19  # Planck constant in keV·s
    c = 299792458       # Speed of light in m/s
    return h * c / energy_kev

def phase_retrieval(I, delta_beta, dist_obj_detector, energy_kev, pixel_size):
    """
    Apply phase retrieval according to formula (6) from Weitkamp et al. (2011).

    Parameters
    ----------
    I : cupy.ndarray
        Measured intensity image.
    delta_beta : float
        Beta/delta ratio.
    dist_obj_detector : float
        Propagation distance (m).
    energy_kev : float
        X-ray energy (keV).
    pixel_size : float
        Detector pixel size (m).

    Returns
    -------
    phi : numpy.ndarray
        Reconstructed phase map.
    """
    wavelength = keVtoLambda(energy_kev)
    ny, nx = I.shape

    # Spatial frequencies
    u = fftfreq(nx, d=pixel_size)
    v = fftfreq(ny, d=pixel_size)
    U, V = cp.meshgrid(u, v, indexing='ij')

    # Fourier transform of the image
    I_fft = fftshift(fft2(I))

    # Compute filter
    denominator = 1 + cp.pi * wavelength * dist_obj_detector * delta_beta * (U**2 + V**2)
    denominator[denominator == 0] = cp.finfo(float).eps  # Avoid division by zero

    # Apply filter and inverse FFT
    phi_fft = I_fft / denominator

    phi = - cp.log(cp.maximum(cp.real(ifft2(ifftshift(phi_fft))), cp.finfo(cp.float32).eps)) * delta_beta * 0.5

    return phi

def get_padding_size(image, energy, effective_pixel_size, distance):
    """
    Calculate the padding size for a 2D image.

    Parameters
    ----------
    image : cupy.ndarray
        2D array of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).

    Returns
    -------
    tuple
        Padded sizes (nx_padded, ny_padded)
    """
    ny, nx = image.shape
    wavelength = keVtoLambda(energy)

    n_margin = math.ceil(3 * wavelength * distance / (2 * effective_pixel_size ** 2))
    nx_margin = nx + 2 * n_margin
    ny_margin = ny + 2 * n_margin

    nx_padded = int(2 ** math.ceil(math.log2(nx_margin)))
    ny_padded = int(2 ** math.ceil(math.log2(ny_margin)))

    return nx_padded, ny_padded

def padding(image, energy, effective_pixel_size, distance):
    """
    Pad a 2D image to avoid edge artifacts during phase retrieval with the closest value.

    Parameters
    ----------
    image : cupy.ndarray
        2D array of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).

    Returns
    -------
    tuple
        Padded image, padded nx, padded ny
    """
    ny, nx = image.shape
    nx_padded, ny_padded = get_padding_size(image, energy, effective_pixel_size, distance)

    top = (ny_padded - ny) // 2
    bottom = ny_padded - ny - top
    left = (nx_padded - nx) // 2
    right = nx_padded - nx - left

    padded_image = cp.pad(image, ((top, bottom), (left, right)), mode='reflect')
    return padded_image, nx_padded, ny_padded

def unsharp_mask(image, sigma, coeff):
    blurred = gaussian_filter(image, sigma=sigma, mode='reflect')
    return (1 + coeff) * image - coeff * blurred

def clean_outliers_cupy(image_gpu, threshold=3, size=3):
    # image_gpu doit être un tableau CuPy (image transférée sur le GPU)
    
    mean = cp.mean(image_gpu)
    std = cp.std(image_gpu)
    
    # Calcul des bornes d'acceptabilité
    lower = mean - threshold * std
    upper = mean + threshold * std

    # Détection des outliers
    outliers = (image_gpu < lower) | (image_gpu > upper)

    # Calcul de la médiane locale
    median_img = median_filter(image_gpu, size=size)

    # Création de la copie nettoyée
    cleaned_image = image_gpu.copy()
    cleaned_image[outliers] = median_img[outliers]

    return cleaned_image


def process_projection(proj, nx, ny, energy, effective_pixel_size, distance, delta_beta, pixel_size):
    """
    Process a single projection image.

    Parameters
    ----------
    proj : cupy.ndarray
        Projection image.
    nx : int
        Original width of the image.
    ny : int
        Original height of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).
    delta_beta : float
        Ratio beta/delta for phase retrieval.
    pixel_size : float
        Detector pixel size (m).

    Returns
    -------
    numpy.ndarray
        Cropped phase-retrieved projection.
    """

    proj_cp = cp.asarray(proj)
    padded_proj, nx_padded, ny_padded = padding(proj_cp, energy, effective_pixel_size, distance)
    retrieved_proj = phase_retrieval(padded_proj, delta_beta, distance, energy, pixel_size)
    retrieved_proj = clean_outliers_cupy(retrieved_proj, threshold=3, size=3)

    x_margin = (nx_padded - nx) // 2
    y_margin = (ny_padded - ny) // 2

    return retrieved_proj[y_margin:y_margin + ny, x_margin:x_margin + nx].get()

def double_flatfield_correction(projs):
    """
    Apply double flat-field correction to an image.

    Parameters
    ----------
    proj : 2D numpy array
        Image to correct

    Returns
    -------
    I_corr : 2D numpy array
        Corrected image

    """
    mean_proj = np.mean(projs, axis=0)

    mean_proj[mean_proj == 0] = 1e-6

    I_corr = projs / mean_proj

    return I_corr

def apply_left_weighting(projs, CoR):
    """
    Applique un poids linéaire sur la partie gauche des projections.
    """
    weights = np.linspace(0, 1, CoR)[None, None, :]
    projs[:, :, :CoR] *= weights

    return projs 
   
def create_sinogram_slice(projs, CoR, slice_idx):
    """
    Create a sinogram from a set of projections.
    """
    theta, _, ny = projs.shape

    sino = np.zeros((theta//2, 2 * ny - CoR))

    flip = projs[:theta // 2, slice_idx, ::-1]  # np.flip optimisé

    sino[:, :ny] += flip
    sino[:,  -ny:] += projs[theta//2:, slice_idx, :]

    return sino

def create_sinogram(projs, CoR):
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

def create_angles(sinogram, end=pi):
    """
    Create angles for a sinogram.

    Parameters
    ----------
    sinogram : numpy.ndarray
        2D or 3D array representing the sinogram(s).
    end : float
        The end angle for the angles array (default is pi, can be 2*pi).

    Returns
    -------
    angles : numpy.ndarray
        Array of angles in radians.
    """
    if sinogram.ndim == 2:
        num_angles = sinogram.shape[0]  # Nombre d'angles pour un sinogramme 2D
    elif sinogram.ndim == 3:
        num_angles = sinogram.shape[1]  # Nombre d'angles pour un sinogramme 3D
    else:
        raise ValueError("Sinogram must be 2D or 3D.")

    angles = np.linspace(0, end, num_angles, endpoint=False)
    return angles

def reconstruct_from_sinogram_slice(sinogram, angles):
    """
    Reconstruct a 2D image from a sinogram using FBP_CUDA algorithm from ASTRA Toolbox.

    Parameters:
    - sinogram: 2D numpy array (angles, detectors) containing the sinogram.
    - angles: 1D numpy array of rotation angles (in radians).

    Returns:
    - reconstruction: 2D numpy array representing the reconstructed image.
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

def create_disk_mask(sinogram):
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

def preprocess(experiment, viewer, widget):
    """
    Apply flatfield and darkfield corrections to the sample layers 
    using the data stored in the experiment object.
    """
    sample_layer = viewer.layers[experiment.sample_images].data

    # Adjust for axis order changes in Napari
    sample_layer = np.transpose(sample_layer, viewer.dims.order)

    if experiment.darkfield is not None and experiment.flatfield is not None:
        print("Applying flat and dark corrections")
        darkfield_layer = np.mean(viewer.layers[widget.darkfield_selection.currentText()].data, axis=0)
        flatfield_layer = np.mean(viewer.layers[widget.flatfield_selection.currentText()].data, axis=0)
        sample_layer = (sample_layer - darkfield_layer) / (flatfield_layer - darkfield_layer)

    elif experiment.darkfield is not None:
        print("Applying dark correction")
        darkfield_layer = np.mean(viewer.layers[widget.darkfield_selection.currentText()].data, axis=0)
        sample_layer = sample_layer - darkfield_layer

    elif experiment.flatfield is not None:
        print("Applying flat correction")
        flatfield_layer = np.mean(viewer.layers[widget.flatfield_selection.currentText()].data, axis=0)
        sample_layer = sample_layer / flatfield_layer

    return {'preprocess' : sample_layer}

def paganin_filter(projs, pixel_size, effective_pixel_size, distance, energy, delta_beta):
    """
    Apply Paganin filter to a set of projections.

    Parameters
    ----------
    projs : cupy.ndarray
        3D array of projection images.
    pixel_size : float
        Detector pixel size (m).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).
    energy : float
        Energy of the X-ray beam (keV).
    delta_beta : float
        Ratio beta/delta for phase retrieval.
    sigma : float
        Standard deviation for Gaussian filter.
    coeff : float
        Coefficient for unsharp mask.
    
    Returns
    -------
    projs : cupy.ndarray
        3D array of phase-retrieved projections.
    """

    retrieved_projs = np.zeros(projs.shape, dtype=np.float32)
    for i in tqdm(range(projs.shape[0]), desc='Processing Paganin'):
        retrieved_projs[i] = process_projection(projs[i], projs.shape[2], projs.shape[1], energy, effective_pixel_size, distance, delta_beta, pixel_size)

    return {'paganin': retrieved_projs}