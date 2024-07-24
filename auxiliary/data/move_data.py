import os
import sys

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from dataset_ht import HtDataset
from auxiliary.utils.colors import bcolors as c

import re
import shutil


def move_feature_files(verbose=1):
    """
    Auxiliary function to move data from Daniela's extraction to the data folder.
    :param verbose:
    :return:
    """

    ds = HtDataset()

    feature_path = '/run/user/1003/gvfs/smb-share:server=tierra.cnic.es,share=sc/LAB_MT/RESULTADOS/DanielaV/MASTER/EXTRACTION/'

    for group in ds.specimens.keys():
        for specimen in ds.specimens[group]:
            f_raw = os.path.join(feature_path, f'2019{specimen}_cell_properties_radiomics.csv')
            f_out = os.path.join(ds.data_path, group, 'Features', f'2019{specimen}_cell_properties_radiomics.csv')

            if os.path.exists(f_raw):
                if verbose:
                    print(f'{c.OKGREEN}Found{c.ENDC}: {f_raw}')
                shutil.move(f_raw, f_out)
            else:
                if verbose:
                    print(f'{c.FAIL}Not found{c.ENDC}: {f_raw}')


if __name__ == '__main__':
    move_feature_files()
