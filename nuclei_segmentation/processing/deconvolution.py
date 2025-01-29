"""
Advanced 3D Confocal Deconvolution with VarPSF (Poisson MLE) in TensorFlow
--------------------------------------------------------------------------
Features:
 1) Gibson–Lanni style PSF generation (simplified)
 2) Depth-dependent varPSF (blockwise approach)
 3) Poisson MLE with i-divergence (Kullback-Leibler) as the cost
 4) Tikhonov (Laplacian) regularization
 5) GPU-accelerated 3D FFT via TensorFlow
 6) Background offset handling
"""

import os
import sys

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir)))

from util.data import imaging
from util import values as v


class MicroscopeParameters:
    """
    Holds confocal microscope parameters:
      - microscope_type: e.g. "Confocal"
      - NA: numerical aperture
      - excitation_wavelength_nm
      - emission_wavelength_nm
      - lens_immersion_refr_index
      - num_excitation_photons: approximate photon budget
    """
    def __init__(
            self,
            microscope_type='Confocal',
            NA=1.3,
            excitation_wavelength_nm=488.0,
            emission_wavelength_nm=520.0,
            lens_immersion_refr_index=1.474, # Glycerin
            num_excitation_photons=1
    ):
        self.microscope_type = microscope_type
        self.NA = NA
        self.excitation_wavelength_nm = excitation_wavelength_nm
        self.emission_wavelength_nm = emission_wavelength_nm
        self.lens_immersion_refr_index = lens_immersion_refr_index
        self.num_excitation_photons = num_excitation_photons


def generate_confocal_psf(
    z_size: int,
    y_size: int,
    x_size: int,
    dz_um: float,
    dy_um: float,
    dx_um: float,
    depth_um: float,
    micro_params: MicroscopeParameters
) -> np.ndarray:
    """
    Generate a simplified 3D confocal PSF via a Gaussian + Airy ring approach.

    We approximate the effect of excitation & emission:
      - The final PSF lateral size scales ~ 0.4*(emission_wavelength/NA).
      - A small difference from the excitation wavelength is introduced to
        mimic confocal effects (some references use sqrt(excitation*emission)).

    Without sample RI or pinhole specifics, this is a generic approximation.

    Parameters
    ----------
    z_size, y_size, x_size : int
        Output PSF volume shape (Z, Y, X).
    dz_um, dy_um, dx_um : float
        Micron size of each voxel in Z, Y, X.
    depth_um : float
        Imaging depth in the sample (um). For demonstration, we do a small
        'focus shift' or broadening with depth, ignoring sample mismatch.
    micro_params : MicroscopeParameters
        Must specify NA, excitation/emission wavelengths, lens_immersion_refr_index, etc.

    Returns
    -------
    psf_3d : np.ndarray, float32
        3D array of shape (z_size, y_size, x_size), sum=1.
    """
    # Convert nm -> um
    exc_um = micro_params.excitation_wavelength_nm * 1e-3
    emi_um = micro_params.emission_wavelength_nm  * 1e-3

    # Basic confocal lateral FWHM ~ 0.4 * lambda / NA
    # We'll do a small weighting of excitation & emission
    wave_um = 0.5*(exc_um + emi_um)  # or sqrt(exc*emi), etc.
    lateral_fwhm = 0.4 * wave_um / micro_params.NA

    # Axial FWHM ~ 1.4 * lambda / NA^2 (rough confocal formula)
    axial_fwhm = 1.4 * wave_um / (micro_params.NA**2)

    # Depth factor: Just a small broadening if depth_um is large
    # (in reality, we'd need sample refr. index mismatch, etc.)
    depth_factor = 1.0 + 0.001 * depth_um
    axial_fwhm *= depth_factor

    # Convert FWHM -> Gaussian sigma
    sig_xy = lateral_fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))
    sig_z = axial_fwhm / (2.0*np.sqrt(2.0*np.log(2.0)))

    # Grid
    zc = z_size//2
    yc = y_size//2
    xc = x_size//2
    zvals = (np.arange(z_size) - zc)*dz_um
    yvals = (np.arange(y_size) - yc)*dy_um
    xvals = (np.arange(x_size) - xc)*dx_um
    zz, yy, xx = np.meshgrid(zvals, yvals, xvals, indexing='ij')

    # Gaussian core
    gauss = np.exp(-0.5 * ((xx / sig_xy) ** 2 + (yy / sig_xy) ** 2 + (zz / sig_z) ** 2))

    # A small ring structure to mimic an Airy pattern
    # This is purely illustrative
    r2 = xx**2 + yy**2
    k = 2.0*np.pi / wave_um
    airy = 0.05 * np.cos(k * np.sqrt(r2 + (0.8*zz)**2))**2

    psf_3d = gauss*(1.0 + airy)
    psf_3d[psf_3d < 0] = 0
    psf_3d /= psf_3d.sum()
    return psf_3d.astype(np.float32)


@tf.function
def fft_convolve_3d(volume: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """
    Circular convolution in 3D via rfft3d / irfft3d.
    volume, kernel: shape (Z, Y, X), float32
    Returns same shape as 'volume'.
    """
    vol_fft = tf.signal.rfft3d(volume)
    ker_fft = tf.signal.rfft3d(kernel)
    prod = vol_fft * ker_fft
    result = tf.signal.irfft3d(prod, volume.shape)
    return result


@tf.function
def fft_convolve_3d_same(vol: tf.Tensor, ker: tf.Tensor) -> tf.Tensor:
    zv, yv, xv = vol.shape
    zk, yk, xk = ker.shape

    # Pad to a shape that can hold both
    padZ = max(zv, zk)
    padY = max(yv, yk)
    padX = max(xv, xk)

    vol_pad = tf.pad(vol, [[0, padZ - zv],
                           [0, padY - yv],
                           [0, padX - xv]])
    ker_pad = tf.pad(ker, [[0, padZ - zk],
                           [0, padY - yk],
                           [0, padX - xk]])

    vol_fft = tf.signal.rfft3d(vol_pad)
    ker_fft = tf.signal.rfft3d(ker_pad)
    product = vol_fft * ker_fft
    conv_pad = tf.signal.irfft3d(product, vol_pad.shape)

    # Crop back to the volume shape (zv, yv, xv)
    out = conv_pad[:zv, :yv, :xv]
    return out


@tf.function
def laplacian_3d(vol: tf.Tensor) -> tf.Tensor:
    lap = -6.*vol
    lap += tf.roll(vol, shift=1, axis=0) + tf.roll(vol, shift=-1, axis=0)
    lap += tf.roll(vol, shift=1, axis=1) + tf.roll(vol, shift=-1, axis=1)
    lap += tf.roll(vol, shift=1, axis=2) + tf.roll(vol, shift=-1, axis=2)
    return lap


@tf.function
def i_divergence(measured: tf.Tensor, estimate: tf.Tensor, eps=1e-12) -> tf.Tensor:
    """
    I-divergence ~ sum( m * log(m / e) - (m - e) ), ignoring negative/zero values.
    """
    m_clip = tf.maximum(measured, eps)
    e_clip = tf.maximum(estimate, eps)
    return tf.reduce_sum(m_clip * tf.math.log(m_clip/e_clip) - (m_clip - e_clip))


def varpsf_rl_iteration(
    estimate: tf.Tensor,
    measured: tf.Tensor,
    z_chunk_slices,
    psf_list,
    alpha=1e-4,
    eps=1e-12
) -> tf.Tensor:
    """
    Perform one global iteration for varPSF:
      For each Z-chunk, do a local RL update with chunk's PSF + Tikhonov.
    """
    new_est = tf.identity(estimate)

    for i, (zs, ze) in enumerate(z_chunk_slices):
        sub_est = new_est[zs:ze, :, :]
        sub_meas = measured[zs:ze, :, :]
        psf = psf_list[i]

        # forward
        # reblur = fft_convolve_3d(sub_est, psf)
        reblur = fft_convolve_3d_same(sub_est, psf)
        ratio = sub_meas / (reblur + eps)
        # backward
        correction = fft_convolve_3d_same(ratio, psf)

        # RL update
        sub_update = sub_est * correction

        # Tikhonov
        lap = laplacian_3d(sub_update)
        sub_update = sub_update - alpha*lap
        sub_update = tf.maximum(sub_update, 0.0)

        # Insert back
        new_est = tf.concat([
            new_est[:zs, :, :],
            sub_update,
            new_est[ze:, :, :]
        ], axis=0)

    return new_est


def varpsf_deconvolution(
    raw_data_np: np.ndarray,
    z_chunk_um: float,
    voxel_size=(0.3,0.1,0.1),
    micro_params=MicroscopeParameters(),
    alpha=1e-4,
    num_iters=10,
    verbose=0
) -> np.ndarray:
    """
    varPSF deconvolution using the specified confocal parameters (Poisson MLE).
    raw_data_np: 3D data array (Z, Y, X)
    z_chunk_um:  approximate chunk thickness in microns
    voxel_size:  (dz, dy, dx) in microns
    micro_params: user-defined confocal parameters
    alpha: Tikhonov weight
    num_iters: number of global varPSF iterations
    """
    shapeZ, shapeY, shapeX = raw_data_np.shape
    dz_um, dy_um, dx_um = voxel_size

    # TF Tensors
    measured_tf = tf.constant(raw_data_np, dtype=tf.float32)

    # Build chunk slices
    slices_per_chunk = max(1, int(np.round(z_chunk_um / dz_um)))
    print(f"Chunk size: {slices_per_chunk} slices")
    z_chunk_slices = []
    start = 0
    while start < shapeZ:
        end = min(shapeZ, start + slices_per_chunk)
        z_chunk_slices.append((start, end))
        start = end

    # Precompute PSF for each chunk
    psf_list = []
    for (zs, ze) in z_chunk_slices:
        mid_z = 0.5*(zs+ze)
        depth_um = mid_z*dz_um
        z_sub_size = (ze - zs) + 16
        psf_np = generate_confocal_psf(
            z_size=z_sub_size,
            y_size=shapeY+16,
            x_size=shapeX+16,
            dz_um=dz_um,
            dy_um=dy_um,
            dx_um=dx_um,
            depth_um=depth_um,
            micro_params=micro_params
        )

        # -------------- PLOT --------------
        plot_psf_3d(psf_np, voxel_size=(dz_um, dy_um, dx_um),
                    title=f"Chunk PSF (zs={zs}, ze={ze})")
        # ----------------------------------

        psf_tf = tf.constant(psf_np, dtype=tf.float32)
        psf_list.append(psf_tf)

    # Initialize estimate (e.g. same as measured)
    estimate_tf = tf.Variable(measured_tf + 1e-6)

    # Iterate
    for it in range(num_iters):
        estimate_tf.assign(
            varpsf_rl_iteration(
                estimate_tf, measured_tf,
                z_chunk_slices, psf_list,
                alpha=alpha
            )
        )

        # Monitor i-divergence chunk by chunk
        kl_sum = 0.0
        for i, (zs, ze) in enumerate(z_chunk_slices):
            sub_est = estimate_tf[zs:ze, :, :]
            sub_psf = psf_list[i]
            # forward
            reblur = fft_convolve_3d_same(sub_est, sub_psf)
            kl_val = i_divergence(measured_tf[zs:ze, :, :], reblur)
            kl_sum += kl_val.numpy()

        if verbose:
            print(f"Iteration {it+1}/{num_iters}, i-divergence = {kl_sum:.3f}")

    return estimate_tf.numpy()


def plot_psf_3d(psf_3d, voxel_size=(0.3, 0.1, 0.1), title="PSF"):
    """
    Display mid-slices (XY and XZ) of a 3D PSF with scale in microns (if known).
    psf_3d: shape (Z, Y, X).
    voxel_size: (dz, dy, dx) in microns.
    """
    dz, dy, dx = voxel_size
    zsize, ysize, xsize = psf_3d.shape

    zmid = zsize // 2
    ymid = ysize // 2

    # XY slice at mid-Z
    psf_xy = psf_3d[zmid, :, :]
    # XZ slice at mid-Y (shape (Z, X))
    psf_xz = psf_3d[:, ymid, :]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # XY
    ax0 = axs[0]
    # if you want real size: extent = [0, xsize*dx, ysize*dy, 0]
    ax0.imshow(psf_xy, cmap='inferno', origin='upper')
    ax0.set_title(f"{title}: XY (z={zmid})")
    ax0.set_xlabel("X (px)")
    ax0.set_ylabel("Y (px)")

    # XZ
    ax1 = axs[1]
    ax1.imshow(psf_xz, cmap='inferno', origin='upper')
    ax1.set_title(f"{title}: XZ (y={ymid})")
    ax1.set_xlabel("X (px)")
    ax1.set_ylabel("Z (px)")

    plt.tight_layout()
    plt.show()


def main():

    def deconvolve(raw_data, voxel_size, micro_params, chunk_um=8.0):
        shapeZ, shapeY, shapeX = raw_data.shape

        result = varpsf_deconvolution(
            raw_data,
            z_chunk_um=chunk_um,
            voxel_size=voxel_size,
            micro_params=micro_params,
            alpha=1e-4,
            num_iters=50
        )

        # 8.6 Visualize a mid-Z slice
        z_mid = shapeZ // 2
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        axs[0].imshow(raw_data[z_mid], cmap='hot')
        axs[0].set_title(f"Raw (mid Z)")
        axs[0].axis('off')

        axs[1].imshow(result[z_mid], cmap='hot')
        axs[1].set_title("varPSF Deconvolved (mid Z)")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

        return result

    img_path = v.data_path + 'auxiliary/cova/RawImages/test_interpolacion.tif'
    raw_data = imaging.read_image(img_path, axes='ZYX')
    metadata, _ = imaging.load_metadata(img_path)

    voxel_size = (
        metadata['z_res'],
        metadata['y_res'],
        metadata['x_res']
    )

    print(raw_data.shape)
    print(voxel_size)
    print(raw_data.dtype)

    micro_params = MicroscopeParameters(
        microscope_type="Confocal",
        NA=0.75,
        excitation_wavelength_nm=405.0,
        emission_wavelength_nm=444.0,
        lens_immersion_refr_index=1.518,
        num_excitation_photons=1
    )

    # Make tf use CPU
    with tf.device('/CPU:0'):
        result = deconvolve(raw_data, voxel_size, micro_params)

    imaging.save_nii(
        result.astype(np.float32),
        img_path.replace('.tif', '_deconvolved.nii.gz'),
        axes='ZYX'
    )


if __name__ == '__main__':
    main()