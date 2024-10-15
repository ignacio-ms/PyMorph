import numpy as np

from scipy import ndimage
from skimage import exposure, morphology
from skimage.restoration import denoise_bilateral
import cv2

from auxiliary.data import imaging
from auxiliary.utils.colors import bcolors as c


def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1.,1.,1.), option=1):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        # warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    for ii in np.arange(1,niter):

        # calculate the diffs
        deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]

        # update the image
        stackout += gamma*(UD+NS+EW)

    return stackout


def reconstruct(img, **kwargs):
    assert kwargs['metadata'] is not None, 'Metadata is required for isotropy reconstruction step.'

    metadata = kwargs['metadata']
    resampling_factor = metadata['x_res'] / metadata['z_res']
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
        }

        if pipeline is None:
            pipeline = [
                'isotropy',
                'norm_percentile',
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
            cv2.normalize(img[z], None, 0, 1, cv2.NORM_MINMAX)
            for z in range(img.shape[0])
        ])

    @staticmethod
    def norm_adaptative(img):
        return np.array([
            exposure.equalize_adapthist(img[z], clip_limit=0.03)
            for z in range(img.shape[0])
        ])

    @staticmethod
    def norm_percentile(img, **kwargs):
        default_kwargs = {
            'low': 5,
            'high': 95
        }

        default_kwargs.update(kwargs)
        low, high = np.percentile(img, (default_kwargs['low'], default_kwargs['high']))

        return np.array([
            exposure.rescale_intensity(img[z], in_range=(low, high))
            for z in range(img.shape[0])
        ])


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
            'sigma_color': 0.1,
            'sigma_spatial': 10
        }

        default_kwargs.update(kwargs)
        return np.array([
            denoise_bilateral(img[z], **default_kwargs)
            for z in range(img.shape[0])
        ])

    @staticmethod
    def isotropy(img, **kwargs):
        assert kwargs['metadata'] is not None, 'Metadata is required for isotropy step.'

        metadata = kwargs['metadata']
        resampling_factor = metadata['z_res'] / metadata['x_res']

        if 'verbose' in kwargs:
            print(
                f'{c.OKBLUE}Image resolution{c.ENDC}: \n'
                f'X: {metadata["x_res"]} um/px\n'
                f'Y: {metadata["y_res"]} um/px\n'
                f'Z: {metadata["z_res"]} um/px'
            )
            print(f'Resampling factor: {resampling_factor}')

        return ndimage.zoom(img, (resampling_factor, 1, 1), order=0)

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
        img = imaging.read_image(img_path, axes='ZYX', verbose=verbose)
        metadata, _ = imaging.load_metadata(img_path)

        for step in self.pipeline:
            print(f'{c.OKGREEN}Running step{c.ENDC}: {step}')

            step_func = self.mapped_pipeline[step]
            step_kwargs = self.filter_kwargs(step_func, kwargs)

            if step in ['isotropy']:
                img = step_func(img, metadata=metadata, **step_kwargs)
                continue

            img = step_func(img, **step_kwargs)

        if test_name:
            imaging.save_nii(
                img, img_path.replace('.nii.gz', f'_{test_name}_filtered.nii.gz'),
                verbose=verbose, axes='ZYX'
            )

        return img.astype(np.int8)

