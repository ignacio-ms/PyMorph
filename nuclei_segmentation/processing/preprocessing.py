import numpy as np

from scipy import ndimage
from skimage import exposure, morphology
from skimage.transform import rescale
from skimage.restoration import denoise_bilateral
import cv2

from util.data import imaging
from util.misc.colors import bcolors as c
from nuclei_segmentation.processing.denoising import anisodiff3, denoise_cellpose
from nuclei_segmentation.processing.intensity_calibration import (
    compute_z_profile_no_mask, compute_z_profile,
    logistic_inverted, fit_inverted_logistic,
    correction_factor
)


def reconstruct(img, **kwargs):
    assert kwargs['metadata'] is not None, 'Metadata is required for isotropy reconstruction step.'

    metadata = kwargs['metadata']
    resampling_factor = metadata['x_res'] / metadata['z_res']
    print(f'{c.OKBLUE}Reconstructing image{c.ENDC}: {resampling_factor}')
    return ndimage.zoom(img, (resampling_factor, 1, 1), order=0)


class Preprocessing:
    def __init__(self, pipeline=None):
        self.mapped_pipeline = {
            'norm_minmax': self.norm_minmax,
            'norm_adaptative': self.norm_adaptative,
            'norm_percentile': self.norm_percentile,
            'equalization': self.equalize,
            'anisodiff': self.anisodiff,
            'bilateral': self.bilateral,
            'isotropy': self.isotropy,
            'gaussian': self.gaussian,
            'median': self.median,
            'gamma': self.gamma,
            'rescale_intensity': self.rescale_intensity,
            'intensity_calibration': self.intensity_calibration,
            'cellpose_denoising': self.cellpose_denoising,
        }

        if pipeline is None:
            pipeline = [
                'intensity_calibration',
                'isotropy',
                'norm_percentile',
                'cellpose_denoising',
                'bilateral',
            ]

        assert all(step in self.mapped_pipeline for step in pipeline), 'Invalid pipeline step.'
        self.pipeline = pipeline

    @staticmethod
    def filter_kwargs(func, kwargs):
        """
        Filter `kwargs` to only pass arguments that `func` accepts.
        """
        import inspect
        sig = inspect.signature(func)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return filtered_kwargs

    @staticmethod
    def norm_minmax(img):
        return np.array([
            cv2.normalize(img[z], None, 0.0, 1.0, cv2.NORM_MINMAX)
            for z in range(img.shape[0])
        ])

    @staticmethod
    def norm_adaptative(img):
        return np.array([
            exposure.equalize_adapthist(img[z], clip_limit=0.05)
            for z in range(img.shape[0])
        ])
        # return exposure.equalize_adapthist(img, clip_limit=0.05, kernel_size=(5, 5, 5))

    @staticmethod
    def norm_percentile(img, **kwargs):
        default_kwargs = {
            'low': 0,
            'high': 100
        }

        default_kwargs.update(kwargs)
        low, high = np.percentile(img, (default_kwargs['low'], default_kwargs['high']))

        # return np.array([
        #     exposure.rescale_intensity(img[z], in_range=(low, high))
        #     for z in range(img.shape[0])
        # ])
        return exposure.rescale_intensity(
            1.0 * img,
            in_range=(low, high), out_range=(0.0, 1.0)
        )


    @staticmethod
    def equalize(img):
        return exposure.equalize_hist(img)

    @staticmethod
    def anisodiff(img, **kwargs):
        """
        kwargs:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every iteration
        """
        default_kwargs = {
            'niter': 3,
            'kappa': 40,
            'gamma': 0.1,
            'step': (1., 1., 1.),
            'option': 1,
        }

        default_kwargs.update(kwargs)
        return anisodiff3(img, **default_kwargs)

    @staticmethod
    def bilateral(img, **kwargs):
        default_kwargs = {
            'win_size': 3,
            'sigma_color': .1,
            'sigma_spatial': 10
        }

        default_kwargs.update(kwargs)

        if img.ndim == 2:
            return denoise_bilateral(img, **default_kwargs)

        return np.array([
            denoise_bilateral(img[z], **default_kwargs)
            for z in range(img.shape[0])
        ])

    @staticmethod
    def isotropy(img, **kwargs):
        assert kwargs['metadata'] is not None, 'Metadata is required for isotropy step.'

        if 'image' in kwargs:
            img = kwargs['image']

        metadata = kwargs['metadata']
        resampling_factor = metadata['z_res'] / metadata['x_res']
        # img_iso = ndimage.zoom(
        #     img, (resampling_factor, 1, 1),
        #     order=3, mode='nearest', prefilter=True
        # )

        img_iso = rescale(
            img, (resampling_factor, 1, 1),
            order=3, mode='edge',
            clip=True,
            anti_aliasing=False,
            preserve_range=True
        )

        if 'verbose' in kwargs:
            print(
                f'\t{c.OKBLUE}Image resolution{c.ENDC}: \n'
                f'\tX: {metadata["x_res"]} um/px\n'
                f'\tY: {metadata["y_res"]} um/px\n'
                f'\tZ: {metadata["z_res"]} um/px'
            )
            print(f'\tResampling factor: {resampling_factor}')
            print(f'\tOriginal shape: {img.shape}')
            print(f'\tResampled shape: {img_iso.shape}')

        return img_iso

    @staticmethod
    def intensity_calibration(img, **kwargs):
        default_kwargs = {
            'mask': None,
            'p0': None,
            'maxfev': 50000,
            'z_ref': 0,
        }
        default_kwargs.update(kwargs)

        # Images are passed as ZYX; we want to transpose to XYZ
        img = np.swapaxes(img, 0, 2)
        z_slices, z_indices = img.shape[2], np.arange(img.shape[2])

        # Compute the intensity profile along the z-axis
        zprofile = compute_z_profile_no_mask if default_kwargs['mask'] is None else compute_z_profile
        fg_means, bg_means = zprofile(img)

        # Fit inverted logistic function
        popt = fit_inverted_logistic(
            z_indices, fg_means,
            p0=default_kwargs['p0'], maxfev=default_kwargs['maxfev']
        )
        L, U, k, x0 = popt

        if 'verbose' in kwargs:
            print(f'{c.OKBLUE}Intensity calibration fitted parameters{c.ENDC}:')
            print(f'\tL: {L}\n\tU: {U}\n\tk: {k}\n\tx0: {x0}')

        # Calibrate image
        calibration_factors = [
            correction_factor(z, popt, z_ref=default_kwargs['z_ref'])
            for z in z_indices
        ]

        img_calibrated = np.zeros_like(img, dtype=img.dtype)
        for z, factor in zip(range(z_slices), calibration_factors):
            img_calibrated[..., z] = img[..., z] * factor

        # Transpose back to ZYX
        return np.swapaxes(img_calibrated, 0, 2)

    @staticmethod
    def cellpose_denoising(img, **kwargs):
        default_kwargs = {
            'diameter': 17,
            'channels': [0, 0],
        }
        default_kwargs.update(kwargs)

        return denoise_cellpose(img, **kwargs)

    @staticmethod
    def gaussian(img, **kwargs):
        default_kwargs = {'sigma': 0.8}
        default_kwargs.update(kwargs)
        return ndimage.gaussian_filter(img, **kwargs)

    @staticmethod
    def median(img, **kwargs):
        default_kwargs = {'size': 3}
        default_kwargs.update(kwargs)
        return ndimage.median_filter(img, **kwargs)

    @staticmethod
    def gamma(img, **kwargs):
        default_kwargs = {'gamma': 0.5}
        default_kwargs.update(kwargs)
        return exposure.adjust_gamma(img, **kwargs)

    @staticmethod
    def rescale_intensity(img, **kwargs):
        default_kwargs = {
            'in_range': (np.percentile(img, 5), np.percentile(img, 95)),
            'out_range': (0, 1)
        }
        default_kwargs.update(kwargs)
        return exposure.rescale_intensity(img, **kwargs)

    def run(self, img_path, test_name=None, verbose=0, **kwargs):
        try:
            img = kwargs['image']
            assert img is not None
        except Exception:
            img = imaging.read_image(img_path, axes='ZYX', verbose=verbose)
        img = img.astype(np.uint16)
        metadata, _ = imaging.load_metadata(img_path)

        for step in self.pipeline:
            print(f'{c.OKGREEN}Running pro-processing step{c.ENDC}: {step}')

            step_func = self.mapped_pipeline[step]
            step_kwargs = self.filter_kwargs(step_func, kwargs)

            if step in ['isotropy']:
                img = step_func(img, metadata=metadata, verbose=verbose, **step_kwargs)
                continue

            elif step in ['intensity_calibration']:
                img = step_func(img, verbose=verbose, **step_kwargs)
                continue

            img = step_func(img, **step_kwargs)

        if test_name:
            imaging.save_nii(
                img, img_path.replace('.nii.gz', f'_{test_name}_filtered.nii.gz'),
                verbose=verbose, axes='ZYX'
            )

        imaging.save_nii(
            img,
            img_path.replace('.nii.gz', f'_preprocessed.nii.gz').replace('.tif', f'_preprocessed.nii.gz'),
            verbose=verbose, axes='ZYX'
        )

        return img.astype(np.float16)

