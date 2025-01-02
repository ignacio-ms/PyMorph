#!/bin/bash

for i in {1..9}
do
    rm -r /run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/Gr${i}/3DShape/Tissue/myocardium/cell_map/*
done
