from qtpy.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel
import math
from cupy.fft import fft2, ifft2
from numpy import pi
import numpy as np
import gc
from joblib import Parallel, delayed
from tqdm import tqdm
from skimage.draw import disk
import astra
import cupy as cp

BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e+2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


def _wavelength(energy):
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy

def paganin_filter(
        data, pixel_size, dist, energy, db, W, pad):
    """
    Perform single-step phase retrieval from phase-contrast measurements
    :cite:`Paganin:02`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    pixel_size : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter for Paganin method.
    method : string
        phase retrieval method. Standard Paganin or Generalized Paganin.
    db : float, optional
        delta/beta for generalized Paganin phase retrieval 
    W :  float
        Characteristic transverse lenght scale    	
    pad : bool, optional
        If True, extend the size of the projections by padding with zeros.
    Returns
    -------
    ndarray
        Approximated 3D tomographic phase data.
    """

    # New dimensions and pad value after padding.
    py, pz, val = _calc_pad(data, pixel_size, dist, energy, pad)

    # Compute the reciprocal grid.
    dx, dy, dz = data.shape
    print('Generalized Paganin method')
    kf = _reciprocal_gridG(pixel_size, dy + 2 * py, dz + 2 * pz)
    phase_filter = cp.fft.fftshift(
        _paganin_filter_factorG(energy, dist, kf, pixel_size, db, W))
    prj = cp.full((dy + 2 * py, dz + 2 * pz), val, dtype=data.dtype)
    _retrieve_phase(data, phase_filter, py, pz, prj, pad)

    return -db * data * 0.5


def _retrieve_phase(data, phase_filter, px, py, prj, pad):
    dx, dy, dz = data.shape
    num_jobs = data.shape[0]
    normalized_phase_filter = phase_filter / phase_filter.max()

    for m in tqdm(range(num_jobs), desc='Retrieving phase'):
        prj[px:dy + px, py:dz + py] = cp.asarray(data[m], dtype=data.dtype)
        prj[:px] = prj[px]
        prj[-px:] = prj[-px-1]
        prj[:, :py] = prj[:, py][:, cp.newaxis]
        prj[:, -py:] = prj[:, -py-1][:, cp.newaxis]
        fproj = fft2(prj)
        fproj *= normalized_phase_filter
        proj = cp.real(ifft2(fproj))
        if pad:
            proj = proj[px:dy + px, py:dz + py]
        data[m] = proj.get()


def _calc_pad(data, pixel_size, dist, energy, pad):
    """
    Calculate new dimensions and pad value after padding.

    Parameters
    ----------
    data : ndarray
        3D tomographic data.
    pixel_size : float
        Detector pixel size in cm.
    dist : float
        Propagation distance of the wavefront in cm.
    energy : float
        Energy of incident wave in keV.
    pad : bool
        If True, extend the size of the projections by padding with zeros.

    Returns
    -------
    int
        Pad amount in projection axis.
    int
        Pad amount in sinogram axis.
    float
        Pad value.
    """
    dx, dy, dz = data.shape
    wavelength = _wavelength(energy)
    py, pz, val = 0, 0, 0
    if pad:
        val = _calc_pad_val(data)
        py = _calc_pad_width(dy, pixel_size, wavelength, dist)
        pz = _calc_pad_width(dz, pixel_size, wavelength, dist)

    return py, pz, val

def _paganin_filter_factorG(energy, dist, kf, pixel_size, db, W):
    """
        Generalized phase retrieval method
        Paganin et al 2020
        diffracting feature ~2*pixel size
    """
    aph = db*(dist*_wavelength(energy))/(4*PI)
    return 1 / (1.0 - (2*aph/(W**2))*(kf-2))


def _calc_pad_width(dim, pixel_size, wavelength, dist):
    pad_pix = cp.ceil(PI * wavelength * dist / pixel_size ** 2)
    return int((pow(2, cp.ceil(cp.log2(dim + pad_pix))) - dim) * 0.5)


def _calc_pad_val(data):
    return float(cp.mean((data[..., 0] + data[..., -1]) * 0.5))

def _reciprocal_gridG(pixel_size, nx, ny):
    """
    Calculate reciprocal grid for Generalized Paganin method.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    nx, ny : int
        Size of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Considering diffracting feature ~2*pixel size
    # Sampling in reciprocal space.
    indx = cp.cos(_reciprocal_coord(pixel_size, nx)*2*PI*pixel_size)
    indy = cp.cos(_reciprocal_coord(pixel_size, ny)*2*PI*pixel_size)
    idx, idy = cp.meshgrid(indy, indx)
    return idx + idy


def _reciprocal_coord(pixel_size, num_grid):
    """
    Calculate reciprocal grid coordinates for a given pixel size
    and discretization.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    num_grid : int
        Size of the reciprocal grid.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    n = num_grid - 1
    rc = cp.arange(-n, num_grid, 2, dtype=cp.float32)
    rc *= 0.5 / (n * pixel_size)
    return rc

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
        delta = experiment.delta
        beta = experiment.beta
        dist_object_detector = experiment.dist_object_detector

        proj = paganin_filter(sample_layer, 
            pixel_size=pixel_size * 1e2, dist=dist_object_detector * 1e2, energy=energy,  
            db=delta/beta, W=5*pixel_size*1e2, pad=True)
        
        add_image_to_layer({"Reconstruction": proj}, "FBP", viewer)

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
        delta = experiment.delta
        beta = experiment.beta
        dist_object_detector = experiment.dist_object_detector

        projs = paganin_filter(sample_layer, 
            pixel_size=pixel_size * 1e2, dist=dist_object_detector * 1e2, energy=energy,  
            db=delta/beta, W=3*pixel_size*1e2, pad=True)

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