import os
import numpy as np
import h5py
import nibabel as nib
import sys
import json

sys.path.insert(1, "/homedtic/dvarela")

import util_daniela as u

f = open("/homedtic/dvarela/specimens.json")
data = json.load(f)


def convert_nii_rotate(fileNII, fileH5):
    """."""
    img_crop_zyx = u.read_nii(fileNII, cellpose=True)
    print(img_crop_zyx.shape)
    hf = h5py.File(fileH5, "a")
    _ = hf.create_dataset("raw", data=img_crop_zyx)
    hf.close()
    print(fileH5)


if __name__ == "__main__":
    flatten_list = [
        element
        for sublist in [data[i] for i in ["stage1", "stage2", "stage3", "stage4"]]
        for element in sublist
    ]
    mems = "/homedtic/dvarela/CardiacRegion/all/mem"
    for sp in flatten_list:
        file_nii = os.path.join(mems, f"2019{sp}_mGFP_CardiacRegion_0.5.nii.gz")
        if os.path.isfile(file_nii):
            print(f"RUNNING {file_nii}")
            file_h5 = file_nii.replace(".nii.gz", "_ZXY.h5")
            CR_MEM = nib.load(file_nii).get_fdata()
            CR_MEM = CR_MEM[:, :, :, 0]
            print(CR_MEM.shape)
            array_zxy = np.swapaxes(np.swapaxes(CR_MEM, 0, 2), 1, 2)
            print(array_zxy.shape)
            hf = h5py.File(file_h5, "a")
            dset = hf.create_dataset("raw", data=array_zxy)
            hf.close()
            print("SAVED H5")
        else:
            print(f"NOT FOUUUNDD!!!! {file_nii}")
    # mem = "/homedtic/dvarela/DECON_05/MGFP/20190208_E2_mGFP_decon_0.5.nii.gz"
    # CR_MEM = nib.load(mem).get_fdata()
    # CR_MEM = CR_MEM[:, :, :, 0]
    # print(CR_MEM.shape)
    # array_zxy = np.swapaxes(np.swapaxes(CR_MEM, 0, 2), 1, 2)
    # print(array_zxy.shape)
    # save_path = "/homedtic/dvarela/DECON_05/MGFP/20190208_E2_mGFP_decon_0.5_ZXY.H5"
    # hf = h5py.File(save_path, "a")
    # dset = hf.create_dataset("raw", data=array_zxy)
    # hf.close()
    # print("SAVED H5")
    # folder = "/homedtic/dvarela/DECON_05/MGFP"

    # especimens = [
    #     "20190504_E1",
    #     "20190404_E2",
    #     "20190520_E4",
    #     "20190516_E3",
    #     "20190806_E3",
    #     "20190520_E2",
    #     "20190401_E3",
    #     "20190517_E1",
    #     "20190520_E1",
    #     "20190401_E1",
    # ]
    # files = [
    #     os.path.join(
    #         os.path.join("/homedtic/dvarela/CardiacRegion", e),
    #         f"{e}_mGFP_CardiacRegion_0.5.nii.gz",
    #     )
    #     for e in especimens
    # ]
