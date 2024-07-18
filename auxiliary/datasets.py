# Standard packages
import os
import re

# Custom packages
from auxiliary import values as v
from auxiliary.colors import bcolors as c


class HtDataset:
    def __init__(self):
        self.data_path = v.data_path
        self.specimens = v.specimens

        self.raw_nuclei_path = []
        self.raw_membrane_path = []

        self.seg_nuclei_path = []
        self.seg_membrane_path = []

        self.missing_nuclei = []
        self.missing_membrane = []

    def get_img_paths(self, type='Segmentation', verbose=0):
        """
        Get paths for both Nuclei and Membrane images.
        :param type: Type of images to get. (Segmentation, RawImages)
        """
        if type not in ['Segmentation', 'RawImages']:
            raise ValueError(f'Invalid type: {type} (Segmentation, RawImages)')

        for group in self.specimens.keys():
            for level in ['Nuclei', 'Membrane']:
                try:
                    f_raw_dir = os.path.join(self.data_path, group, 'Segmentation', level)
                    walk = os.walk(f_raw_dir).__next__()
                except StopIteration:
                    if verbose:
                        print(f'{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                    continue

                spec_set = set(self.specimens[group])
                for img in walk[2]:
                    for specimen in spec_set:
                        if re.search(specimen, img):
                            if level == 'Nuclei':
                                if type == 'RawImages':
                                    self.raw_nuclei_path.append(os.path.join(f_raw_dir, img))
                                else:
                                    self.seg_nuclei_path.append(os.path.join(f_raw_dir, img))
                            else:
                                if type == 'RawImages':
                                    self.raw_membrane_path.append(os.path.join(f_raw_dir, img))
                                else:
                                    self.seg_membrane_path.append(os.path.join(f_raw_dir, img))
                            spec_set.remove(specimen)
                            break

        if verbose:
            if type == 'RawImages':
                print(f'{c.OKGREEN}Raw Images - Nuclei{c.ENDC}: {len(self.raw_nuclei_path)}')
                print(f'{c.OKGREEN}Raw Images - Membrane{c.ENDC}: {len(self.raw_membrane_path)}')
            else:
                print(f'{c.OKGREEN}Segmented Nuclei{c.ENDC}: {len(self.seg_nuclei_path)}')
                print(f'{c.OKGREEN}Segmented Membrane{c.ENDC}: {len(self.seg_membrane_path)}')

    def check_specimens(self, verbose=0):
        """
        Check if both Nuclei and Membrane segmentation images are available for each specimen.
        Get a list of not segmented images. (missing_nuclei, missing_membrane)
        :param verbose: Verbosity level.
        """
        todo_membrane = []
        todo_nuclei = []

        for group in self.specimens.keys():
            if verbose:
                print(f'{c.OKBLUE}Group{c.ENDC}: {group}')

            for level in ['Nuclei', 'Membrane']:
                if verbose:
                    print(f'{c.BOLD}Level{c.ENDC}: {level}')

                try:
                    f_raw_dir = os.path.join(self.data_path, group, 'Segmentation', level)
                    walk = os.walk(f_raw_dir).__next__()
                except StopIteration:
                    if verbose:
                        print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                    continue

                spec_set = set(self.specimens[group])

                for img in walk[2]:
                    for specimen in spec_set:
                        if re.search(specimen, img):
                            if verbose:
                                print(f'\t{c.OKGREEN}Found{c.ENDC}: {img}')
                            spec_set.remove(specimen)
                            break

                for i in spec_set:
                    if verbose:
                        print(f'\t{c.FAIL}Missing{c.ENDC}: {i}')
                    aux = os.path.join(
                        self.data_path, group, 'RawImages', level,
                        f'2019{i}_DAPI_decon_0.5.nii.gz' if level == 'Nuclei' else f'2019{i}_mGFP_decon_0.5.nii.gz'
                    )

                    if os.path.exists(aux):
                        if level == 'Nuclei':
                            todo_nuclei.append(aux)
                        else:
                            todo_membrane.append(aux)
                    else:
                        if verbose:
                            print(f'\t{c.FAIL}No raw image: {aux}{c.ENDC}')

        self.missing_nuclei = todo_nuclei
        self.missing_membrane = todo_membrane

    def get_raw_img_paths(self):
        """
        Get paths for both Nuclei and Membrane raw images.
        :return: List of paths. (raw_nuclei_path, raw_membrane_path)
        """

        if len(self.raw_nuclei_path) == 0 and len(self.raw_membrane_path) == 0:
            self.get_img_paths(type='RawImages')
            return self.raw_nuclei_path, self.raw_membrane_path

        return self.raw_nuclei_path, self.raw_membrane_path

    def get_seg_img_paths(self):
        """
        Get paths for both Nuclei and Membrane segmented images.
        :return: List of paths. (seg_nuclei_path, seg_membrane_path)
        """

        if len(self.seg_nuclei_path) == 0 and len(self.seg_membrane_path) == 0:
            self.get_img_paths(type='Segmentation')
            return self.seg_nuclei_path, self.seg_membrane_path

        return self.seg_nuclei_path, self.seg_membrane_path

    def get_missing_img_paths(self):
        """
        Get paths for both Nuclei and Membrane without segmentation images.
        :return: List of paths. (missing_nuclei, missing_membrane)
        """

        if len(self.missing_nuclei) == 0 and len(self.missing_membrane) == 0:
            self.check_specimens()
            return self.missing_nuclei, self.missing_membrane

        return self.missing_nuclei, self.missing_membrane
