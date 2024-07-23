import os
import numpy as np
import h5py
import nibabel as nib
import sys
import json


if __name__ == "__main__":
    f = open("/homedtic/dvarela/specimens.json")
    data = json.load(f)
    flatten_list = [
        element
        for sublist in [data[i] for i in ["stage1", "stage2", "stage3", "stage4"]]
        for element in sublist
    ]
    mems = "/homedtic/dvarela/CROPS"
    for sp in flatten_list:
        file_h5 = os.path.join(
            mems, f"2019{sp}_mGFP_CardiacRegion_0.5_ZXY_predictions.h5"
        )
        print(f"RUNNING {file_h5}")
        pred_zxy = np.array(h5py.File(file_h5, "r")["predictions"])
        print(pred_zxy.shape)
        pred_xyz = np.swapaxes(np.swapaxes(pred_zxy[0, :, :, :], 0, 2), 0, 1)
        print(pred_xyz.shape)
        ni_img = nib.Nifti1Image(pred_xyz, affine=np.eye(4))
        res = "/homedtic/dvarela/RESULTS/membranes/GASP_PNAS"
        nib.save(
            ni_img,
            os.path.join(
                res, f"2019{sp}_mGFP_CardiacRegion_0.5_XYZ_predictions.nii.gz"
            ),
        )
