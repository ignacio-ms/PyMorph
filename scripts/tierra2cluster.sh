#!/bin/bash

for i in {6..9}
do
    rsync -avhP --progress --no-perms --ignore-times /run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/Gr${i}/Segmentation/Membrane/* imarcoss@10.149.80.48:/data_lab_MT/Ignacio/data/data//Gr${i}/Segmentation/Membrane/
done
