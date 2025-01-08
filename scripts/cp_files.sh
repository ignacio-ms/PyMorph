#!/bin/bash

for i in {1..9}
do
  mkdir -p /run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/auxiliary/features_checkpoint/Gr${i}/Filtered/
  cp /run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/Gr${i}/Features/Filtered/* /run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/auxiliary/features_checkpoint/Gr${i}/Filtered/
done
