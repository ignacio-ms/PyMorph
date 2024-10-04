import pandas as pd

from nuclei_segmentation import my_cellpose as cp
from nuclei_segmentation import preprocessing

from auxiliary.data import imaging
from auxiliary.data.dataset_ht import HtDataset, find_specimen


class ModelTester:
    def __init__(self, model):
        self.model = model

    def run(self, img_path, pipeline, type, stitch_threshold, test_name, verbose=0):
        if verbose:
            print(f'Running test: {test_name}')

        do_3D = True if type == '3D' else False
        img_path_out = (
            img_path
            .replace('.nii.gz', f'_{test_name}.nii.gz')
            .replace('RawImages', 'Segmentation')
        )

        metadata, _ = imaging.load_metadata(img_path)
        img = cp.load_img(
            img_path, pipeline=pipeline,
            test_name=test_name, verbose=verbose
        )

        masks = cp.run(
            self.model, img,
            diameter=17, channels=[0, 0],
            stitch_threshold=stitch_threshold,
            do_3D=do_3D, verbose=verbose
        )
        masks = preprocessing.reconstruct(masks, metadata=metadata)
        imaging.save_nii(masks, img_path_out, axes='ZYX', verbose=verbose)


