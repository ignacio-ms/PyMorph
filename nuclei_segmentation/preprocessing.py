import numpy as np

from scipy import ndimage
from skimage import exposure, morphology
from skimage.restoration import denoise_bilateral
import cv2

from auxiliary.data import imaging


def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1.,1.,1.), option=1, ploton=False):
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
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

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

    # create the plot figure, if requested
    if ploton:
        import pylab as pl
        from time import sleep

        showplane = stack.shape[0]//2

        fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
        ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)

        ax1.imshow(stack[showplane,...].squeeze(),interpolation='nearest')
        ih = ax2.imshow(stackout[showplane,...].squeeze(),interpolation='nearest',animated=True)
        ax1.set_title("Original stack (Z = %i)" %showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

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

        if ploton:
            iterstring = "Iteration %i" %(ii+1)
            ih.set_data(stackout[showplane,...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return stackout


class Preprocessing:
    def __init__(self, pipeline=None):
        self.mapped_pipeline = {
            'normalization': self.normalize,
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
                'normalization',
                'bilateral'
            ]

        assert all(step in self.mapped_pipeline for step in pipeline)
        self.pipeline = pipeline

    @staticmethod
    def normalize(img):
        return np.array([
            cv2.normalize(img[..., z], None, 0, 1, cv2.NORM_MINMAX)
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
        if 'kwargs' not in kwargs:
            kwargs['kwargs'] = {
                'niter': 3,
                'kappa': 40,
                'gamma': 0.1,
                'step': (1., 1., 1.),
                'option': 1,
                'ploton': False
            }

        return anisodiff3(img, **kwargs)

    @staticmethod
    def bilateral(img, **kwargs):
        if 'kwargs' not in kwargs:
            kwargs['kwargs'] = {
                'win_size': 3,
                'sigma_color': 0.1,
                'sigma_spatial': 10
            }

        return np.array([
            denoise_bilateral(img[..., z], **kwargs)
            for z in range(img.shape[0])
        ])

    @staticmethod
    def isotropy(img, **kwargs):
        assert kwargs['metadata'] is not None

        metadata = kwargs['metadata']
        resampling_factor = metadata['z_res'] / metadata['x_res']
        return ndimage.zoom(img, (resampling_factor, 1, 1), order=0)

    @staticmethod
    def reconstruct(img, **kwargs):
        assert kwargs['metadata'] is not None

        metadata = kwargs['metadata']
        resampling_factor = metadata['x_res'] / metadata['z_res']
        return ndimage.zoom(img, (resampling_factor, 1, 1), order=0)

    @staticmethod
    def gaussian(img, **kwargs):
        if 'kwargs' not in kwargs:
            kwargs['kwargs'] = {
                'sigma': .8
            }

        return ndimage.gaussian_filter(img, **kwargs)

    @staticmethod
    def median(img, **kwargs):
        if 'kwargs' not in kwargs:
            kwargs['kwargs'] = {
                'size': 3
            }

        return ndimage.median_filter(img, **kwargs)

    @staticmethod
    def gamma(img, **kwargs):
        if 'kwargs' not in kwargs:
            kwargs['kwargs'] = {
                'gamma': .5
            }

        return exposure.adjust_gamma(img, **kwargs)

    @staticmethod
    def rescale_intensity(img, **kwargs):
        if 'kwargs' not in kwargs:
            kwargs['kwargs'] = {
                'in_range': (np.percentile(img, 5), np.percentile(img, 95)),
                'out_range': (0, 1)
            }

        return exposure.rescale_intensity(img, **kwargs)

    def load(self, img_path, test_name=None, verbose=0, **kwargs):
        img = imaging.read_image(img_path, axes='YZX', verbose=verbose)
        metadata, _ = imaging.load_metadata(img_path)

        for step in self.pipeline:
            if step in ['isotropy']:
                img = self.mapped_pipeline[step](img, metadata=metadata)
                continue

            img = self.mapped_pipeline[step](img, **kwargs)

        if test_name:
            imaging.save_nii(
                img, img_path.replace('.nii.gz', f'{test_name}_filtered.nii.gz'),
                verbose=verbose, axes='ZYX'
            )

        return img

