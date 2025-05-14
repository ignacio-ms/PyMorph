import sys
import os

from rich.progress import Progress

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from util.misc.colors import bcolors as c
from util.data import imaging


os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8/bin/"

BASE_DIR = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/LAB/Ignacio/auxiliary/jyo_ploidy/'
RAW_MEM_DIR = os.path.join(BASE_DIR, 'raw', 'membrane')
RAW_MEM_CROPS_DIR = os.path.join(BASE_DIR, 'raw', 'membrane', 'crops')
RAW_NUC_DIR = os.path.join(BASE_DIR, 'raw', 'nuclei')
RAW_NUC_CROPS_DIR = os.path.join(BASE_DIR, 'raw', 'nuclei', 'crops')

MEM_SEG_DIR = os.path.join(BASE_DIR, 'segmentation', 'membrane')
NUCLEI_SEG_DIR = os.path.join(BASE_DIR, 'segmentation', 'nuclei')

CROP_SIZE = 300 # Size of the cropped image (300x300xZ)
CROP_MID = CROP_SIZE // 2 # Midpoint of the crop size


def crop_image_to_tiles(img, out_path, do_2D=True):
    """
    Crop the image into different tiles and save them as separate files. If the image is 3D, it will be cropped
    in 3D tiles (300x300xZ). If the image is 2D, it will be cropped in 2D tiles (300x300). If the image is
    3D and do_2D is True, it will be cropped in 2D tiles (300x300) and saved as separate files; <filename>_<tile_id>_<tile_Z>.tif
    or <filename>_<tile_id>.tif.
    CROP_SIZE is the size of the cropped image (300x300xZ) in xy, never in Z.
    :param img: Image to be cropped.
    :param out_path: Path where the cropped images will be saved.
    :param do_2D: If True, return the crops in 2D. (Default: True)
    :return: Cropped image.
    """

    x, y, z = img.shape
    num_tiles_x = x // CROP_SIZE
    num_tiles_y = y // CROP_SIZE
    num_tiles_z = z if do_2D else 1

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    n_tiles = 0
    with Progress() as progress:
        task = progress.add_task(f"Cropping image", total=num_tiles_x * num_tiles_y * num_tiles_z)
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                for k in range(num_tiles_z):
                    x_start = i * CROP_SIZE
                    y_start = j * CROP_SIZE
                    z_start = k

                    x_end = min(x_start + CROP_SIZE, x)
                    y_end = min(y_start + CROP_SIZE, y)
                    z_end = min(z_start + 1, z) if do_2D else z

                    tile = img[x_start:x_end, y_start:y_end, z_start:z_end]

                    if do_2D:
                        tile_path = os.path.join(out_path, f"{os.path.basename(img)}_{i}_{j}.tif")
                    else:
                        tile_path = os.path.join(out_path, f"{os.path.basename(img)}_{i}_{j}_{k}.tif")

                    imaging.save_tiff_imagej_compatible(tile_path, tile, axes='XYZ')
                    n_tiles += 1

    print(f"Total number of tiles: {n_tiles}")


def crop_images():
    """
    Crop both membrane and nuclei images into different tiles and save them as separate files.
    :return: None
    """

    if not os.path.exists(RAW_MEM_CROPS_DIR):
        os.makedirs(RAW_MEM_CROPS_DIR)

    for img_path in os.listdir(RAW_MEM_DIR):
        try:
            if not img_path.endswith('.tif'):
                continue

            print(f"{c.OKBLUE}Cropping image{c.ENDC}: {img_path}")
            print(f"{c.OKBLUE}Membrane... {c.ENDC}")
            img = imaging.read_image(os.path.join(RAW_MEM_DIR, img_path), axes='XYZ')
            crop_image_to_tiles(img, os.path.join(RAW_MEM_CROPS_DIR, img_path), do_2D=True)

            print(f"{c.OKBLUE}Nuclei... {c.ENDC}")
            img = imaging.read_image(os.path.join(RAW_NUC_DIR, img_path), axes='XYZ')
            crop_image_to_tiles(img, os.path.join(RAW_NUC_CROPS_DIR, img_path), do_2D=True)

            print(f"{c.OKGREEN}Cropped image{c.ENDC}: {img_path}")

        except Exception as e:
            print(f"{c.FAIL}Error cropping image: {img_path}{c.ENDC}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == '__main__':
    crop_images()

