import networkx as nx
import numpy as np
from skimage import morphology, measure
from skimage.segmentation import watershed, find_boundaries
from scipy import ndimage as ndi
from skimage.morphology import erosion, dilation, ball
from skimage.feature import peak_local_max

from utils.data import imaging
from utils.data.dataset_ht import find_specimen, HtDataset
from utils.misc.colors import bcolors as c


class PostProcessing:
    def __init__(self, pipeline=None):
        self.mapped_pipeline = {
            'remove_small_objects': self.remove_small_objects,
            'split': self.split,
            'merge_connected_components': self.merge_connected_components,
            'merge_graph': self.merge_graph,
            'clean_boundaries_opening': self.opening,
            'clean_boundaries_closing': self.closing,
            'remove_large_objects': self.remove_large_objects,
        }

        if pipeline is None:
            pipeline = ['remove_small_objects', 'remove_large_objects']

        assert all(step in self.mapped_pipeline for step in pipeline)
        self.pipeline = pipeline

    @staticmethod
    def filter_kwargs(func, kwargs):
        import inspect
        sig = inspect.signature(func)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return filtered_kwargs

    @staticmethod
    def remove_small_objects(segmentation, percentile=5, verbose=0):
        """
        Remove small objects from segmentation based on a computed size threshold.

        Parameters:
        - segmentation: numpy array, the segmented image.
        - percentile: float, the percentile to determine the size threshold.

        Returns:
        - numpy array, the segmentation with small objects removed.
        """
        labeled_segmentation = measure.label(segmentation)
        component_sizes = np.bincount(labeled_segmentation.ravel())

        # Exclude background (label 0)
        component_sizes[0] = 0

        size_threshold = np.percentile(component_sizes[component_sizes > 0], percentile)

        small_object_labels = np.where(component_sizes < size_threshold)[0]
        small_object_mask = np.isin(labeled_segmentation, small_object_labels)

        segmentation[small_object_mask] = 0

        if verbose:
            print(f'Removed {len(small_object_labels)} small objects.')

        return segmentation

    @staticmethod
    def remove_large_objects(segmentation, percentile=95, verbose=0):
        """
        Remove large objects from segmentation based on a computed size threshold.

        Parameters:
        - segmentation: numpy array, the segmented image.
        - percentile: float, the percentile to determine the size threshold.

        Returns:
        - numpy array, the segmentation with large objects removed.
        """
        labeled_segmentation = measure.label(segmentation)
        component_sizes = np.bincount(labeled_segmentation.ravel())

        # Exclude background (label 0)
        component_sizes[0] = 0

        size_threshold = np.percentile(component_sizes[component_sizes > 0], percentile)

        large_object_labels = np.where(component_sizes > size_threshold)[0]
        large_object_mask = np.isin(labeled_segmentation, large_object_labels)

        segmentation[large_object_mask] = 0

        if verbose:
            print(f'Removed {len(large_object_labels)} large objects.')

        return segmentation

    @staticmethod
    def split(segmentation, min_size=500, connectivity=2):
        """
        Split cells in the segmentation using 3D connected component analysis (CCA).

        Parameters:
            segmentation (np.ndarray): 3D array representing the segmented mask.
            min_size (int): Minimum volume of regions to keep after splitting.

        Returns:
            np.ndarray: Segmentation with cells split into distinct components.
        """

        # Create a copy of the original segmentation
        split_labels = np.zeros_like(segmentation, dtype=np.int16)

        # Loop over each unique label (cell) in the segmentation
        unique_labels = np.unique(segmentation)

        # Start labeling from 1
        current_label = 1

        for label in unique_labels:
            if label == 0:
                # Skip background
                continue

            # Extract the region corresponding to the current label
            cell_region = (segmentation == label)

            # Perform connected component analysis on the cell region to split it
            labeled_cell, num_labels = measure.label(cell_region, connectivity=connectivity, return_num=True)

            # Optionally, filter out small regions based on volume
            regions = measure.regionprops(labeled_cell)
            for region in regions:
                if region.area >= min_size:
                    # Assign a new unique label to each connected component (split cells)
                    split_labels[labeled_cell == region.label] = current_label
                    current_label += 1

        return split_labels

    @staticmethod
    def merge_connected_components(segmentation, **kwargs):
        default_kwargs = {
            'connectivity': 2,
            'min_volume': 500,
            'dilation_size': 9  # Extend the dilation for finding neighbors across z-axis
        }
        default_kwargs.update(kwargs)

        merged_labels = segmentation.copy()

        # Perform connected component analysis in 3D
        labeled_image, num_labels = measure.label(merged_labels, return_num=True,
                                                  connectivity=default_kwargs['connectivity'])

        # Calculate the volume (number of voxels) for each labeled region
        regions = measure.regionprops(labeled_image)

        # Create a new array to store the modified labels
        merged_labels = labeled_image.copy()

        # Create a set to track which regions have been merged
        merged_set = set()

        for region in regions:
            # If the region's volume is less than the minimum volume, consider it for merging
            if region.area < default_kwargs['min_volume']:
                # Skip regions that have already been merged
                if region.label in merged_set:
                    continue

                # Get the coordinates of the small region's voxels
                coords = region.coords

                # Dilate in 3D to find neighboring regions across z-slices
                dilated_region = morphology.dilation(labeled_image == region.label,
                                                     morphology.ball(default_kwargs['dilation_size']))

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
                        # Skip already merged regions
                        if neighbor_label in merged_set:
                            continue
                        neighbor_region = next(r for r in regions if r.label == neighbor_label)
                        if neighbor_region.area > largest_size:
                            largest_neighbor = neighbor_label
                            largest_size = neighbor_region.area

                    if largest_neighbor is not None:
                        # Assign the small region's voxels to the largest neighboring region
                        for coord in coords:
                            merged_labels[tuple(coord)] = largest_neighbor

                        # Mark the small region and the neighbor as merged
                        merged_set.add(region.label)
                        merged_set.add(largest_neighbor)

        return merged_labels

    @staticmethod
    def merge_graph(segmented_volume, max_distance=5):
        """
        Link cells across slices using a graph-based approach.

        Parameters:
        - segmented_volume: 3D numpy array (Z, Y, X) with labeled segments
        - max_distance: Maximum centroid distance to consider for linking cells

        Returns:
        - linked_volume: 3D numpy array with linked labels across slices
        """
        # default_kwargs = {
        #     'max_distance': 10,
        # }
        # default_kwargs.update(kwargs)

        num_slices = segmented_volume.shape[0]
        G = nx.Graph()

        segmented_volume = segmented_volume.astype(np.int32)

        # Dictionary to store cell properties for each slice
        cells_in_slices = {}

        # Build nodes for each cell in each slice
        for z in range(num_slices):
            current_slice = segmented_volume[z]
            props = measure.regionprops(current_slice)
            cells_in_slices[z] = []

            for prop in props:
                if prop.label == 0:
                    continue  # Skip background

                # Store centroid and label information
                cell_info = {
                    'slice': z,
                    'label': prop.label,
                    'centroid': prop.centroid,
                    'area': prop.area
                }
                cells_in_slices[z].append(cell_info)

                # Add node to the graph
                node_id = (z, prop.label)
                G.add_node(node_id, **cell_info)

        # Add edges between cells in adjacent slices
        for z in range(num_slices - 1):
            cells_current = cells_in_slices[z]
            cells_next = cells_in_slices[z + 1]

            for cell_curr in cells_current:
                for cell_next in cells_next:
                    # Calculate Euclidean distance between centroids
                    distance = np.linalg.norm(np.array(cell_curr['centroid']) - np.array(cell_next['centroid']))

                    if distance <= max_distance:
                        # Add an edge with the distance as weight
                        node_curr = (cell_curr['slice'], cell_curr['label'])
                        node_next = (cell_next['slice'], cell_next['label'])
                        G.add_edge(node_curr, node_next, weight=distance)

        # Find connected components in the graph
        connected_components = list(nx.connected_components(G))

        # Assign new labels based on connected components
        linked_volume = np.zeros_like(segmented_volume)
        new_label = 1

        for component in connected_components:
            for node in component:
                z, original_label = node
                # Update the labels in the linked volume
                linked_volume[z][segmented_volume[z] == original_label] = new_label
            new_label += 1

        print(f"Number of cells after linking: {new_label - 1}")

        return linked_volume

    @staticmethod
    def opening(segmentation, erosion_radius=3, dilation_radius=3):
        """
        Apply morphological operations to clean cell boundaries.

        Parameters:
        segmentation (np.ndarray): 3D array representing the segmented mask.
        erosion_radius (int): Radius for erosion operation.
        dilation_radius (int): Radius for dilation operation.

        Returns:
        np.ndarray: Segmentation with cleaned boundaries.
        """

        eroded = erosion(segmentation, ball(erosion_radius))
        cleaned = dilation(eroded, ball(dilation_radius))
        return cleaned

    @staticmethod
    def closing(segmentation, erosion_radius=3, dilation_radius=3):
        """
        Apply morphological operations to clean cell boundaries.

        Parameters:
        segmentation (np.ndarray): 3D array representing the segmented mask.
        erosion_radius (int): Radius for erosion operation.
        dilation_radius (int): Radius for dilation operation.

        Returns:
        np.ndarray: Segmentation with cleaned boundaries.
        """

        dilated = dilation(segmentation, ball(dilation_radius))
        cleaned = erosion(dilated, ball(erosion_radius))
        return cleaned

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
            print(f'{c.OKGREEN}Running post-processing step{c.ENDC}: {step}')

            step_func = self.mapped_pipeline[step]
            step_kwargs = self.filter_kwargs(step_func, kwargs)
            img = step_func(img, **step_kwargs)
        return img
