import sys
import os

_backend = 'cupy'

if _backend == 'cupy':
    import cupy as np
    import cupyx.scipy.ndimage as ndimage
else:
    import numpy as np
    from scipy import ndimage


def read_image(path):
    """
    Read an image from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    if path.endswith('.nii.gz'):
        import nibabel as nib
        proxy = nib.load(path)
        img = proxy.get_fdata()
        metadata = proxy.header

        if img.ndim == 4:
            img = img[..., 0]
            if img.shape[-1] > 150:
                img = img[..., :150]

    elif path.endswith('.tif') or path.endswith('.tiff'):
        from tifffile import imread, TiffFile
        img = np.swapaxes(imread(path), 0, 2)

        with TiffFile(path) as tif:
            metadata = tif.pages[0].tags
            metadata = {tag.name: tag.value for tag in metadata.values()}

            resolution = metadata.get('XResolution'), metadata.get('YResolution')
            print(resolution)
            if resolution:
                metadata['x_res'] = 1.0 / resolution[0][0] if resolution[0][0] != 0 else None
                metadata['y_res'] = 1.0 / resolution[1][0] if resolution[1][0] != 0 else None
                metadata['z_res'] = 1.9998574

                metadata['x_res'] *= 1e6
                metadata['y_res'] *= 1e6
            else:
                metadata['x_res'] = metadata.get('XResolution', 1.0)
                metadata['y_res'] = metadata.get('YResolution', 1.0)
                metadata['z_res'] = 1.0

    else:
        raise ValueError(f"Unsupported file format: {path}")

    return img, metadata


def isotropy(img, **kwargs):
    assert kwargs['metadata'] is not None, 'Metadata is required for isotropy step.'

    if 'image' in kwargs:
        img = kwargs['image']

    metadata = kwargs['metadata']
    resampling_factor = metadata['z_res'] / metadata['x_res']

    img_iso = ndimage.zoom(
        img, (1, 1, resampling_factor),
        order=3, mode='nearest',
        prefilter=True,
    )

    print(
        f'\tImage resolution: \n'
        f'\tX: {metadata["x_res"]} um/px\n'
        f'\tY: {metadata["y_res"]} um/px\n'
        f'\tZ: {metadata["z_res"]} um/px'
    )
    print(f'\tResampling factor: {resampling_factor}')
    print(f'\tOriginal shape: {img.shape}')
    print(f'\tResampled shape: {img_iso.shape}')

    return img_iso


def bilateral_filter_2d(image, sigma_s=.1, sigma_r=15):
    return ndimage.gaussian_filter(image, sigma=sigma_s, mode='nearest', truncate=3.0)
    # Ensure the image is a floating-point array
    image = image.astype(np.float64)

    # Create a grid of spatial coordinates
    radius = int(3 * sigma_s)  # Typically, 3*sigma_s is used as the filter radius
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]

    # Spatial Gaussian weights
    spatial_gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_s ** 2))

    # Initialize the output image
    output = np.zeros_like(image)
    normalizer = np.zeros_like(image)

    # Iterate over each pixel
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            # Extract the local patch
            patch = image[i - radius:i + radius + 1, j - radius:j + radius + 1]

            # Compute the intensity Gaussian weights
            intensity_gaussian = np.exp(-((patch - image[i, j]) ** 2) / (2 * sigma_r ** 2))

            # Combine the spatial and intensity weights
            weights = spatial_gaussian * intensity_gaussian

            # Normalize weights
            weights /= weights.sum()

            # Compute the filtered pixel value
            output[i, j] = np.sum(patch * weights)
            normalizer[i, j] = weights.sum()

    return output

def bilateral_filter_3d(image, sigma_s=.1, sigma_r=15):
    return np.swapaxes(np.swapaxes(np.asarray([
        bilateral_filter_2d(image[..., z], sigma_s, sigma_r)
        for z in range(image.shape[-1])
    ]), 0, -1), 0, 1)


def otsu_threshold(image):
    """
    Computes Otsu's threshold for a grayscale image.

    Parameters:
        image (ndarray): Input grayscale image (2D array).

    Returns:
        threshold (float): Optimal threshold value.
    """
    # Compute the histogram of the image
    hist, bin_edges = np.histogram(image.ravel(), bins=256, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Total number of pixels
    total_pixels = image.size

    # Cumulative sums and cumulative means
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * bin_centers)

    # Global mean
    global_mean = cumulative_mean[-1] / total_pixels

    # Between-class variance
    numerator = (global_mean * cumulative_sum - cumulative_mean) ** 2
    denominator = cumulative_sum * (total_pixels - cumulative_sum)
    between_class_variance = numerator / (denominator + 1e-10)  # Add epsilon to avoid division by zero

    # Find the maximum of between-class variance
    optimal_idx = np.argmax(between_class_variance)
    optimal_threshold = bin_centers[optimal_idx]

    return optimal_threshold


def tophat(img, **kwargs):
    default_kwargs = {
        'disk_size': 3
    }

    default_kwargs.update(kwargs)
    disk = ndimage.generate_binary_structure(2, 1)
    print(f"Disk shape: {disk.shape}")
    print(f"Disk: {disk}")

    img_tophat = np.swapaxes(np.swapaxes(np.asarray([
        ndimage.white_tophat(
            img[..., z],
            footprint=disk
        )
        for z in range(img.shape[-1])
    ]), 0, 1), 1, 2)

    thr = otsu_threshold(img_tophat)

    init_mask = img_tophat > thr
    mask_closed = np.swapaxes(np.swapaxes(np.asarray([
        ndimage.binary_closing(init_mask[..., z], disk)
        for z in range(init_mask.shape[-1])
    ]), 0, 1), 1, 2)

    return img * mask_closed

def process(path):
    """
    Real all images in the path and and process them. Save the computation times in a CSV file.
    :param path:
    :return:
    """
    import time

    print(f"Processing images in: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    print(f'Backend: {_backend}')

    # Read images
    img, metadata = read_image(path)
    if _backend == 'cupy':
        img = np.asarray(img)
    print(f"Image type: {type(img)}")
    print(f"Image shape: {img.shape}")
    print(metadata)
    metadata = {k: v for k, v in metadata.items() if k in ['x_res', 'y_res', 'z_res']}

    # Process images
    start_time = time.time()
    # img = isotropy(img, metadata=metadata)
    # img = tophat(img, metadata=metadata)
    img = bilateral_filter_3d(img, sigma_s=1, sigma_r=15)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    # Save results
    # df = pd.DataFrame({'Processing Time': [end_time - start_time]})
    # df.to_csv('processing_times.csv', index=False)
    return end_time - start_time


if __name__ == "__main__":
    import pandas as pd

    rows = []

    # Example usage
    path = "/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/Ignacio/BioIT/setup_cuda_torch_cupy/test_imgs/"
    for img_path in os.listdir(path):
        if img_path.endswith('.nii.gz') or img_path.endswith('.tif'):
            dt = process(os.path.join(path, img_path))
            rows.append({'Processing Time': dt})

    df = pd.DataFrame(rows)
    df.to_csv(f'processing_times_{_backend}.csv', index=False)

