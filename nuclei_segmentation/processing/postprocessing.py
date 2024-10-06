import numpy as np
from skimage import morphology, measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.morphology import erosion, dilation, ball


class PostProcessing:
    def __init__(self, pipeline=None):
        self.mapped_pipeline = {
            'remove_small_objects': self.remove_small_objects,
            '3d_connected_component_analysis': self.connected_component_analysis,
            'merge_by_volume': self.merge_by_volume,
            'clean_boundaries_morphology': self.clean_boundaries_morphology,
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
    def remove_small_objects(segmentation, min_size=500):
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

    def merge_by_volume(segmentation, min_volume=1000):
        """
        Merge small segments across z-axis based on volume.

        Parameters:
            segmentation (np.ndarray): 3D array representing the segmented mask.
            min_volume (int): Minimum volume threshold for valid segments.

        Returns:
            np.ndarray: Segmentation with merged small fragments.
        """
        # Perform connected component analysis in 3D
        labeled_image, num_labels = measure.label(segmentation, connectivity=1, return_num=True)

        # Calculate the volume (number of voxels) for each labeled region
        regions = measure.regionprops(labeled_image)

        # Create a new array to store the modified labels
        merged_labels = labeled_image.copy()

        for region in regions:
            # If the region's volume is less than the minimum volume, consider it for merging
            if region.area < min_volume:
                # Get the coordinates of the small region's voxels
                coords = region.coords

                # Dilate in 3D to find neighboring regions across z-slices
                dilated_region = morphology.dilation(labeled_image == region.label, morphology.ball(1))

                # Get the neighboring labels around this small region (in 3D)
                neighbor_labels = np.unique(labeled_image[dilated_region])
                neighbor_labels = neighbor_labels[neighbor_labels != region.label]  # Exclude the small region itself
                neighbor_labels = neighbor_labels[neighbor_labels != 0]  # Exclude background

                # If there are neighboring labels, merge the small region with the largest neighbor
                if len(neighbor_labels) > 0:
                    # Choose the largest neighboring region (by volume)
                    largest_neighbor = None
                    largest_size = 0
                    for neighbor_label in neighbor_labels:
                        neighbor_region = next(r for r in regions if r.label == neighbor_label)
                        if neighbor_region.area > largest_size:
                            largest_neighbor = neighbor_label
                            largest_size = neighbor_region.area

                    # Assign the small region's voxels to the largest neighboring region
                    for coord in coords:
                        merged_labels[tuple(coord)] = largest_neighbor

        return merged_labels.astype(np.int16)

    @staticmethod
    def clean_boundaries_morphology(segmentation, erosion_radius=2, dilation_radius=2):
        """
        Apply morphological operations to clean cell boundaries.

        Parameters:
        segmentation (np.ndarray): 3D array representing the segmented mask.
        erosion_radius (int): Radius for erosion operation.
        dilation_radius (int): Radius for dilation operation.

        Returns:
        np.ndarray: Segmentation with cleaned boundaries.
        """
        # Erode the segmentation to remove small artifacts and separate touching regions
        eroded = erosion(segmentation, ball(erosion_radius))

        # Dilate the eroded segmentation to restore the original size
        cleaned = dilation(eroded, ball(dilation_radius))

        return cleaned.astype(np.int16)

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
