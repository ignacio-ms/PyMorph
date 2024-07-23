import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import nibabel as nib


def convert_nii_h5(file_nii, file_h5, order_dataset="xyz", order_algorithm="zyx"):
    """for pytorch 3d UNET"""
    # also rotates ZYX to XYZ
    img_crop_xyz = nib.load(file_nii).get_fdata()
    print(img_crop_xyz.shape)
    if order_dataset != order_algorithm:
        if order_dataset == "xyz" and order_dataset == "xyz":
            img_crop_zyx = np.swapaxes(img_crop_xyz, 0, 2)
            print(img_crop_zyx.shape)
    hf = h5py.File(file_h5, "a")  # open a hdf5 file
    _ = hf.create_dataset("raw", data=img_crop_zyx)  # write the data to hdf5 file
    hf.close()  # close the hdf5 file
    print("hdf5 file created --> {}".format(file_h5))


def read_nii_XYZ(file, get_ZYX=False, get_ZXY=False):
    """."""
    if get_ZYX:
        array_zyx = np.swapaxes(nib.load(file).get_fdata(), 0, 2)
        print("AXES: zyx")
        return array_zyx
    if get_ZXY:
        array_zxy = np.swapaxes(np.swapaxes(nib.load(file).get_fdata(), 0, 2), 1, 2)
        print("AXES: zxy")
        return array_zxy
    return nib.load(file).get_fdata()


def read_multiple_nii(list_files):
    """."""
    arrays = [read_nii_XYZ(i) for i in list_files]
    return arrays


def save_nii(array, fileNII):  # pylint:disable=invalid-name
    """."""
    imgMy = nib.Nifti1Image(array, np.eye(4))
    if imgMy.get_data_dtype() == np.dtype(np.uint16):
        nib.save(imgMy, fileNII)
        print("Saved {}".format(fileNII))
    else:
        print("Check array type is np.uint16")
        imgMy = nib.Nifti1Image(array.astype("uint16"), np.eye(4))
        nib.save(imgMy, fileNII)
        print("Saved {}".format(fileNII))
