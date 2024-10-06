from nuclei_segmentation import my_cellpose as cp
from nuclei_segmentation.processing import postprocessing, preprocessing

from auxiliary.data import imaging


class ModelTester:
    def __init__(self, model):
        self.model = model

        self.preprocessing_steps = [
            'norm_minmax', 'norm_adaptive', 'norm_percentile', 'equalization',
            'anisodiff', 'bilateral', 'isotropy', 'gaussian', 'median', 'gamma',
            'rescale_intensity'
        ]
        self.postprocessing_steps = [
            'remove_small_objects', '3d_connected_component_analysis',
            'watershed'
        ]

    def split_pipeline(self, pipeline):
        pre_pipeline = [step for step in pipeline if step in self.preprocessing_steps]
        post_pipeline = [step for step in pipeline if step in self.postprocessing_steps]
        return pre_pipeline, post_pipeline

    def run(self, img_path, pipeline, type, stitch_threshold, cellprob_threshold, test_name, verbose=0):
        if verbose:
            print(f'Running test: {test_name}')

        pre_pipeline, post_pipeline = self.split_pipeline(pipeline)

        do_3D = True if type == '3D' else False
        img_path_out = (
            img_path
            .replace('.nii.gz', f'_{test_name}.nii.gz')
            .replace('RawImages', 'Segmentation')
        )

        metadata, _ = imaging.load_metadata(img_path)
        img = cp.load_img(
            img_path, pipeline=pre_pipeline,
            test_name=test_name, verbose=verbose
        )

        masks = cp.run(
            self.model, img,
            diameter=17, channels=[0, 0],
            stitch_threshold=stitch_threshold,
            cellprob_threshold=cellprob_threshold,
            do_3D=do_3D, verbose=verbose
        )
        masks = preprocessing.reconstruct(masks, metadata=metadata)

        postproc = postprocessing.PostProcessing(pipeline=post_pipeline)
        masks = postproc.run(masks, verbose=verbose)

        imaging.save_nii(masks, img_path_out, axes='ZYX', verbose=verbose)


