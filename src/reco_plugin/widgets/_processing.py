from qtpy.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
import math
from scipy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift
from numpy import pi
import numpy as np
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.draw import disk
import astra



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
    h = 6.58211928e-19  # keV.s
    c = 299792458  # m/s
    return h * c / (energy_kev)

def get_padding_size(image, energy, effective_pixel_size, distance):
    """
    Calculate the padding size for a 2D image.

    Parameters
    ----------
    image : numpy.ndarray
        2D array of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).

    Returns
    -------
    n_margin : int
        Padding size.
    """
    ny, nx = image.shape
    wavelength = keVtoLambda(energy)

    # Calculate the padding size
    n_margin = math.ceil(3 * wavelength * distance / (2 * effective_pixel_size**2))
    
    nx_margin = nx + 2 * n_margin
    ny_margin = ny + 2 * n_margin

    nx_padded = 2 ** math.ceil(math.log2(nx_margin))
    ny_padded = 2 ** math.ceil(math.log2(ny_margin))

    return nx_padded, ny_padded

def padding(image, energy, effective_pixel_size, distance):
    """
    Pad a 2D image to avoid edge artifacts during phase retrieval with the closest value.

    Parameters
    ----------
    image : numpy.ndarray
        2D array of the image.
    energy : float
        Energy of the X-ray beam (keV).
    effective_pixel_size : float
        Effective pixel size of the detector (m).
    distance : float
        Distance between the object and the detector (m).

    Returns
    -------
    padded_image : numpy.ndarray
        Padded image.
    """
    
    ny, nx = image.shape
    nx_padded, ny_padded = get_padding_size(image, energy, effective_pixel_size, distance)

    top = (ny_padded - ny) // 2
    bottom = ny_padded - ny - top
    left = (nx_padded - nx) // 2
    right = nx_padded - nx - left

    return np.pad(image, ((top, bottom), (left, right)), mode='reflect'), nx_padded, ny_padded

def paganin_filter(sample_images, energy_kev, pixel_size, delta_beta, dist_object_detector, beta=1e-10):
    """
    Apply the Paganin filter to an image.

    Parameters
    ----------
    sample_images : numpy.ndarray
        Image to filter.
    energy_kev : float
        Energy of the X-ray beam (keV).
    pixel_size : float
        Effective pixel size of the detector (m).
    delta_beta : float
        Delta over beta value.
    dist_object_detector : float
        Distance between the object and the detector (m).
    beta : float
        Beta value for phase retrieval.

    Returns
    -------
    img_thickness : numpy.ndarray
        Filtered image.
    """
    lambda_energy = keVtoLambda(energy_kev)
    pix_size = pixel_size

    waveNumber = 2 * pi / lambda_energy

    mu = 2 * beta * waveNumber

    fftNum = fftshift(fft2(sample_images))
    Nx, Ny = fftNum.shape

    u = fftfreq(Nx, d=pix_size)
    v = fftfreq(Ny, d=pix_size)

    u, v = np.meshgrid(np.arange(0,Nx), np.arange(0,Ny))
    u = u - (Nx/2)
    v = v - (Ny/2)
    u_m = u / (Nx * pix_size) 
    v_m = v / (Ny * pix_size)
    uv_sqrd = np.transpose(u_m**2 + v_m**2)

    denominator = 1 + pi * delta_beta * dist_object_detector * lambda_energy * uv_sqrd
    denominator[denominator == 0] = np.finfo(float).eps

    tmpThickness = ifft2(ifftshift(fftNum / denominator))
    img_thickness = np.real(tmpThickness)
    img_thickness[img_thickness <= 0] = 0.000000000001

    return (-np.log(img_thickness) / mu) * 1e6

def process_projection(proj, nx, ny, energy, effective_pixel_size, distance, beta, delta, pixel_size):
        """
        Process a single projection image.

        Parameters
        ----------
        proj : numpy.ndarray
            Projection image.
        nx : int
            Width of the image.
        ny : int
            Height of the image.
        white : numpy.ndarray
            Whitefield image.
        dark : numpy.ndarray
            Darkfield image.
        energy : float
            Energy of the X-ray beam (keV).
        effective_pixel_size : float
            Effective pixel size of the detector (m).
        distance : float
            Distance between the object and the detector (m).
        beta : float
            Beta value for phase retrieval.
        delta : float
            Delta value for phase retrieval.
        pixel_size : float
            Detector pixel size (m).

        Returns
        -------
        numpy.ndarray
            Processed image.
        """
        padded_proj, ny_padded, nx_padded = padding(proj, energy, effective_pixel_size, distance)
        retrieved_proj = paganin_filter(padded_proj, energy, pixel_size, delta/beta, distance, beta)
        x_margin = abs(nx_padded - nx) // 2
        y_margin = abs(ny_padded - ny) // 2

        return retrieved_proj[x_margin:x_margin+nx, y_margin:y_margin+ny]

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
    disk_mask = np.zeros((sinogram.shape[2], sinogram.shape[2]))
    rr, cc = disk((sinogram.shape[2]//2, sinogram.shape[2]//2), sinogram.shape[2] // 2)
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
        darkfield_layer = np.mean(viewer.layers[experiment.darkfield].data)
        sample_layer = sample_layer - darkfield_layer

    if experiment.flatfield is not None:
        flatfield_layer = np.mean(viewer.layers[experiment.flatfield].data)
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
        viewer.add_image(image.real, name=f"{name}_{method}")

def process_try_paganin(experiment, viewer):

    print(viewer.dims.order)

    processing_dialog = create_processing_dialog(viewer.window.qt_viewer)

    try:
        sample_layer = apply_corrections_one_slice(viewer, experiment)

        nx, ny = sample_layer.shape

        energy = experiment.energy
        pixel_size = experiment.pixel
        delta = experiment.delta
        beta = experiment.beta
        effective_pixel_size = experiment.effective_pixel
        dist_object_detector = experiment.dist_object_detector

        proj = process_projection(sample_layer, nx, ny, energy, pixel_size, dist_object_detector, beta, delta, effective_pixel_size)
        
        add_image_to_layer({"Reconstruction": proj}, "FBP", viewer)

    except Exception as e:
        print(f"Error processing slice: {e}")

    finally:
        processing_dialog.close()

    

def process_all_slices(experiment, viewer):
    processing_dialog = create_processing_dialog(viewer.window.qt_viewer)

    try:
        sample_layer = apply_corrections(viewer, experiment)

        theta, nx, ny = sample_layer.shape

        energy = experiment.energy
        pixel_size = experiment.pixel
        delta = experiment.delta
        beta = experiment.beta
        effective_pixel_size = experiment.effective_pixel
        dist_object_detector = experiment.dist_object_detector

        projs = np.array(Parallel(n_jobs=-1, backend='threading')(
            delayed(process_projection)(proj, nx, ny, energy, pixel_size, dist_object_detector, beta, delta, effective_pixel_size) 
            for proj in tqdm(sample_layer, desc='Processing Paganin filter')))
        
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