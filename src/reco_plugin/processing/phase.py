import math
from cupy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from cupyx.scipy.ndimage import gaussian_filter, median_filter
import numpy as np
from tqdm import tqdm
import cupy as cp

from cupy import pi

from ..constants import PLANCK_J_S, LIGHT_SPEED, KEV_TO_JOULE, PLANCK_keV_S

def keVtoLambda(keV: float) -> float:
    """
    Convert energy in keV to wavelength in meters.
    """
    # E = keV * KEV_TO_JOULE              # en J
    # wavelength_m = (PLANCK_J_S * LIGHT_SPEED) / E  # en m

    wavelength_m = PLANCK_keV_S * LIGHT_SPEED / keV  # en m
    return wavelength_m

def phase_retrieval(I: cp.ndarray, delta_beta: float, distance: float, energy: float, pixel_size: float) -> cp.ndarray:
    """
    Phase retrieval using Paganin's method. 
    Paganin et al. Journal of Microscopy, 2002.
    """
    wavelength = keVtoLambda(energy)  # en m
    pixel_size = keVtoLambda(pixel_size)  # en m
    ny, nx = I.shape
    u = fftfreq(nx, d=pixel_size)  # en m-1
    v = fftfreq(ny, d=pixel_size)  # en m-1
    U, V = cp.meshgrid(u, v, indexing='xy')  # en m-1
    # I_fft = fftshift(fft2(I)) 
    I_fft = fft2(I) 
    denominator = 1 + pi * wavelength * distance * delta_beta * (U**2 + V**2)
    phi_fft = I_fft / denominator
    # img_real = cp.real(ifft2(ifftshift(phi_fft)))  # en m-1
    img_real = cp.real(ifft2(phi_fft))  # en m-1
    return - cp.log(cp.maximum(img_real, cp.finfo(cp.float32).eps)) * delta_beta * 0.5

def get_padding_size(image: cp.ndarray, energy: float, effective_pixel_size: float, distance: float) -> int:
    """
    Get the padding size for the image using ANKAphase formula.
    Weitkamp et al. Journal of Synchrotron Radiation, 2013.
    """
    ny, nx = image.shape
    wavelength = keVtoLambda(energy)  # en m
    n_margin = math.ceil(3 * wavelength * distance / (2 * effective_pixel_size ** 2))  # en pixels
    nx_margin = nx + 2 * n_margin
    ny_margin = ny + 2 * n_margin
    nx_padded = int(2 ** math.ceil(math.log2(nx_margin)))  # en pixels
    ny_padded = int(2 ** math.ceil(math.log2(ny_margin)))  # en pixels
    return nx_padded, ny_padded

def padding(image: cp.ndarray, energy: float, effective_pixel_size: float, distance: float) -> cp.ndarray:
    """
    Padding the image using ANKAphase formula.
    Weitkamp et al. Journal of Synchrotron Radiation, 2013.
    """
    ny, nx = image.shape
    nx_padded, ny_padded = get_padding_size(image, energy, effective_pixel_size, distance)
    top = (ny_padded - ny) // 2
    bottom = ny_padded - ny - top
    left = (nx_padded - nx) // 2
    right = nx_padded - nx - left
    padded_image = cp.pad(image, ((top, bottom), (left, right)), mode='reflect')
    return padded_image, nx_padded, ny_padded

def unsharp_mask(image: cp.ndarray, sigma: float, coeff: float) -> cp.ndarray:
    """
    Apply unsharp mask to the image.
    """
    blurred = gaussian_filter(image, sigma=sigma, mode='reflect')
    return (1 + coeff) * image - coeff * blurred

def clean_outliers(image: cp.ndarray, threshold=3, size=5) -> cp.ndarray:
    """
    Clean outliers in the image using median filter.
    """
    mean = cp.mean(image)
    std = cp.std(image)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    outlier = (image < lower_bound) | (image > upper_bound)
    median_img = median_filter(image, size=size, mode='reflect')
    cleaned_img = image.copy()
    cleaned_img[outlier] = median_img[outlier]
    return cleaned_img

def process_projection(proj: np.ndarray, energy: float,pixel_size: float, effective_pixel_size: float, distance: float, delta_beta: float) -> np.ndarray:
    """
    Process the projection using Paganin's method.
    """
    ny, nx = proj.shape
    proj_cp = cp.asarray(proj)
    padded_proj, nx_padded, ny_padded = padding(proj_cp, energy, effective_pixel_size, distance)
    retrieved_proj = phase_retrieval(padded_proj, delta_beta, distance, energy, pixel_size)
    retrieved_proj = clean_outliers(retrieved_proj, threshold=3, size=5)
    x_margin = (nx_padded - nx) // 2
    y_margin = (ny_padded - ny) // 2
    return retrieved_proj[y_margin:y_margin + ny, x_margin:x_margin + nx].get()

def paganin_filter(projs: np.ndarray, energy: float, pixel_size: float, effective_pixel_size: float, distance: float, delta_beta: float) -> np.ndarray:
    """
    Apply Paganin filter to the projections.
    """
    retrieved_projs = np.zeros(projs.shape, dtype=np.float32)
    for i in tqdm(range(projs.shape[0]), desc='Processing Paganin'):
        retrieved_projs[i] = process_projection(projs[i],
                                energy, 
                                pixel_size, effective_pixel_size, 
                                distance, delta_beta)
    return {'paganin': retrieved_projs}

