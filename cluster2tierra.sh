#!/bin/bash

for i in {1..11}
do
    rsync -avhP --progress --inplace --no-perms imarcoss@10.149.80.48:/data_lab_MT/Ignacio/data/data//Gr${i}/Segmentation/Nuclei/*_mask_myocardium_splanchnic.nii.gz /run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/Gr${i}/Segmentation/Nuclei/
done
