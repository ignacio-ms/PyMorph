import os
import sys


# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset

from meshes.mesh_reconstruction import run


if __name__ == '__main__':

    disk_size = 3
    ds = HtDataset()
    levels = ['Nuclei', 'Membrane']
    tissues = ['myocardium', 'splanchnic']

    for level in levels:
        print(f'{c.OKBLUE}Level{c.ENDC}: {level}')

        for tissue in tissues:
            print(f'{c.OKBLUE}Tissue{c.ENDC}: {tissue}')
            for group in ds.specimens.items():
                print(f'{c.BOLD}Group{c.ENDC}: {group[0]}')

                for specimen in group[1]:
                    print(f'\t{c.BOLD}Specimen{c.ENDC}: {specimen}')

                    try:
                        img_path, _ = ds.read_specimen(
                            specimen, level, 'Segmentation',
                            filtered=True if level == 'Nuclei' else False, verbose=1
                        )
                        img_path_raw, _ = ds.read_specimen(specimen, level, type='RawImages', verbose=1)
                        lines_path, _ = ds.read_line(specimen, verbose=1)
                        path_out = v.data_path + f'/{group[0]}/3DShape/{tissue}/2019{specimen}_{tissue}.ply'

                        run(img_path, path_out, img_path_raw, lines_path, tissue, level, verbose=1)

                    except Exception as e:
                        print(f'{c.FAIL}Error{c.ENDC}: {e}')
                        continue
