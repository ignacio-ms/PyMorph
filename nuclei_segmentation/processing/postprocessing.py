import numpy as np
from skimage import morphology, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi


class PostProcessing:
    def __init__(self, pipeline=None):
        self.mapped_pipeline = {
            'remove_small_objects': self.remove_small_objects,
            '3d_connected_component_analysis': self.connected_component_analysis,
            'watershed': self.apply_watershed
        }

        if pipeline is None:
            pipeline = ['remove_small_objects']

        assert all(step in self.mapped_pipeline for step in pipeline)
        self.pipeline = pipeline

    @staticmethod
    def filter_kwargs(func, kwargs):
        import inspect
        sig = inspect.signature(func)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return filtered_kwargs

    @staticmethod
    def remove_small_objects(segmentation, min_size=400):
        """Remove small objects from segmentation based on a minimum size."""
        return morphology.remove_small_objects(segmentation, min_size=min_size)

    @staticmethod
    def connected_component_analysis(segmentation):
        """
        Perform 3D connected component analysis to merge fragmented parts of cells.

        This method labels connected components in a 3D volume, ensuring that
        fragmented cell parts are merged based on connectivity.

        Parameters:
            segmentation (np.ndarray): 3D array representing the segmentation mask.

        Returns:
            np.ndarray: Relabeled 3D segmentation with connected components.
        """
        labeled_image, num_labels = measure.label(segmentation, connectivity=1, return_num=True)
        return labeled_image.astype(np.int16)

    @staticmethod
    def apply_watershed(segmentation, markers=None):
        """
        Apply watershed segmentation to separate compacted cells in the image.

        Parameters:
            segmentation (np.ndarray): 3D array representing the binary segmentation mask.
            markers (np.ndarray): Optional, precomputed markers for the watershed.

        Returns:
            np.ndarray: Segmentation result after watershed.
        """
        # Compute the distance transform to identify centers of compacted objects
        distance = ndi.distance_transform_edt(segmentation)

        if markers is None:
            # Automatically generate markers based on local maxima of the distance transform
            local_maxi = morphology.local_maxima(distance)
            markers, _ = ndi.label(local_maxi)

        # Apply watershed segmentation
        labels = watershed(-distance, markers, mask=segmentation)

        return labels

    def run(self, img, **kwargs):
        """
        Run the entire post-processing pipeline on the input image (segmentation).

        Parameters:
            img (np.ndarray): Input 3D segmentation image.
            kwargs: Additional parameters for each post-processing step.

        Returns:
            np.ndarray: The processed segmentation image.
        """
        for step in self.pipeline:
            step_func = self.mapped_pipeline[step]
            step_kwargs = self.filter_kwargs(step_func, kwargs)
            img = step_func(img, **step_kwargs)
        return img
