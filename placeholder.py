# Standard libraries
import os
import sys

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from nuclei_segmentation.run_cellpose import run, load_model

from auxiliary.data import imaging
from auxiliary import values as v


img_path = v.data_path + 'Gr4/RawImages/Nuclei/20190806_E4_DAPI_decon_0.5.nii.gz'
img = imaging.read_image(img_path, axes='ZXY', verbose=1)
img_crop = imaging.crop(img, 0, 512, 0, img.shape[1], 0, img.shape[2])
print(img_crop.shape)


model = load_model(model_type='nuclei')

meta = imaging.load_metadata(img_path)
anisotropy = meta['z_res'] / meta['x_res']

masks = run(
    model, img_crop,
    diameter=17, normalize=True,
    channels=[0, 0], anisotropy=anisotropy,
    verbose=1
)
