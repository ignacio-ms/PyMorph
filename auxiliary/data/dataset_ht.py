# Standard packages
import os
import re

import pandas as pd

# Custom packages
from auxiliary import values as v
from auxiliary.utils.colors import bcolors as c


class HtDataset:
    def __init__(self, data_path=None, specimens=None):
        self.data_path = data_path
        if data_path is None:
            self.data_path = v.data_path

        self.specimens = specimens
        if specimens is None:
            self.specimens = v.specimens

        self.raw_nuclei_path = []
        self.raw_membrane_path = []

        self.seg_nuclei_path = []
        self.seg_membrane_path = []

        self.missing_nuclei = []
        self.missing_nuclei_out = []
        self.missing_membrane = []
        self.missing_membrane_out = []

    def read_img_paths(self, type='Segmentation', verbose=0):
        """
        Get paths for both Nuclei and Membrane images.
        :param type: Type of images to get. (Segmentation, RawImages)
        """
        if type not in ['Segmentation', 'RawImages']:
            raise ValueError(f'Invalid type: {type} (Segmentation, RawImages)')

        for group in self.specimens.keys():
            for level in ['Nuclei', 'Membrane']:
                try:
                    f_raw_dir = os.path.join(self.data_path, group, type, level)
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
                print(f'{c.OKGREEN}Segmented - Nuclei{c.ENDC}: {len(self.seg_nuclei_path)}')
                print(f'{c.OKGREEN}Segmented - Membrane{c.ENDC}: {len(self.seg_membrane_path)}')

    def check_segmentation(self, verbose=0):
        """
        Check if both Nuclei and Membrane segmentation images are available for each specimen.
        Get a list of not segmented images. (missing_nuclei, missing_membrane)
        :param verbose: Verbosity level.
        """
        todo_membrane, todo_membrane_out = [], []
        todo_nuclei, todo_nuclei_out = [], []

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
                        # Second condition as a placeholder (REMOVE IT)
                        if re.search(specimen, img):
                            # if level == 'Nuclei' and not re.search('myocardium_splanchnic', img):
                            #     continue
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

                    aux_out = aux.replace('RawImages', 'Segmentation')
                    aux_out = aux_out.replace(
                        '_DAPI_decon_0.5' if level == 'Nuclei' else '_mGFP_decon_0.5',
                        '_mask'
                    )

                    if os.path.exists(aux):
                        if level == 'Nuclei':
                            todo_nuclei_out.append(aux_out)
                            todo_nuclei.append(aux)
                        else:
                            todo_membrane_out.append(aux_out)
                            todo_membrane.append(aux)
                    else:
                        if verbose:
                            print(f'\t{c.FAIL}No raw image: {aux}{c.ENDC}')

        self.missing_nuclei = todo_nuclei
        self.missing_nuclei_out = todo_nuclei_out
        self.missing_membrane = todo_membrane
        self.missing_membrane_out = todo_membrane_out

    def read_specimen(self, spec, level='Nuclei', type='RawImages', filtered=False, verbose=0):
        """
        Read image for a specific specimen.
        :param spec: Specimen to read.
        :param level: Level of the image to read. (Nuclei, Membrane)
        :param type: Type of image to read. (RawImages, Segmentation)
        :param verbose: Verbosity level.
        :return: (Image path, Output path)
        """

        for group in self.specimens.keys():
            try:
                if filtered:
                    f_raw_dir = os.path.join(self.data_path, group, type, level, 'Filtered')
                else:
                    f_raw_dir = os.path.join(self.data_path, group, type, level)
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                # if verbose:
                #     print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            for img in walk[2]:
                if re.search(spec, img) and not img.endswith('.h5'):
                    if verbose:
                        print(f'\n\t{c.OKGREEN}Found{c.ENDC}: {img}')

                    path = os.path.join(f_raw_dir, img)
                    out_path = path.replace('RawImages', 'Segmentation')
                    out_path = out_path.replace(
                        '_DAPI_decon_0.5' if level == 'Nuclei' else '_mGFP_decon_0.5',
                        '_mask'
                    )

                    return path, out_path

        raise FileNotFoundError(f'No specimen found: {spec} (Read specimen) [{level} - {type}]')

    def read_features(self, spec, level='Nuclei', tissue='myocardium', verbose=0):
        """
        Get feature extraction file path for a specific specimen.
        :param spec: Specimen
        :param level: Level of extraction (Nuclei, Membrane, Cell)
        :param tissue: Tissue of extraction (default: myocardium)
        :return: Path to the feature extraction file.
        """
        for group in self.specimens.keys():
            try:
                f_raw_dir = os.path.join(self.data_path, group, 'Features')
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            for file in walk[2]:
                if re.search(spec, file) and re.search(level, file) and re.search(tissue, file):
                    if verbose:
                        print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')

                    return os.path.join(f_raw_dir, file)

        raise FileNotFoundError(f'No specimen found: {spec} (Read features) [{level} - {tissue}]')

    def read_line(self, spec, verbose=0):
        """
        Read line image for a specific specimen.
        :param spec: Specimen to read.
        :param verbose: Verbosity level.
        :return:
        """
        for group in self.specimens.keys():
            try:
                f_raw_dir = os.path.join(self.data_path, group, 'Segmentation', 'LinesTissue')
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            for img in walk[2]:
                if re.search(spec, img):
                    if verbose:
                        print(f'\t{c.OKGREEN}Found{c.ENDC}: {img}')

                    path = os.path.join(f_raw_dir, img)
                    out_path = path.replace('Nuclei', 'Myocardium')
                    out_path = out_path.replace('line','cardiac_region')

                    return path, out_path

        raise FileNotFoundError(f'No specimen found: {spec} (Read Line)')

    def check_features(self, type='NA', tissue='myocardium', verbose=0):
        """
        Check if features have been extracted for each specimen.
        Get a list of not extracted features. (missing_features)
        :param type: Type of features for output path. (Membrane, Nuclei) (Default: NA)
        :param verbose: Verbosity level.
        :return: List of specimens with missing features and their output paths.
        """
        todo_specimens, todo_out_paths = [], []

        for group in self.specimens.keys():
            if verbose:
                print(f'{c.OKBLUE}Group{c.ENDC}: {group}')

            try:
                f_raw_dir = os.path.join(self.data_path, group, 'Features')
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            spec_set = set(self.specimens[group])

            for file in walk[2]:
                for specimen in spec_set:
                    if re.search(specimen, file) and re.search(type, file) and re.search(tissue, file):
                        if verbose:
                            print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')
                        spec_set.remove(specimen)
                        break

            for i in spec_set:
                if verbose:
                    print(f'\t{c.FAIL}Missing features{c.ENDC}: {i}')

                todo_specimens.append(i)
                todo_out_paths.append(
                    os.path.join(f_raw_dir, f'2019{i}_cell_properties_radiomics_{type}_{tissue}.csv')
                )

        return todo_specimens, todo_out_paths

    def check_features_complex(self, type='NA', tissue='myocardium', attr='columnarity', verbose=0):
        """
        Check if complex features have been extracted for each specimen.
        Get a list of not extracted features. (missing_features)
        :param type: Type of features for output path. (Membrane, Nuclei) (Default: NA)
        :param verbose: Verbosity level.
        :return: List of specimens with missing features and their output paths.
        """
        todo_specimens = []

        for group in self.specimens.keys():
            if verbose:
                print(f'{c.OKBLUE}Group{c.ENDC}: {group}')

            try:
                f_raw_dir = os.path.join(self.data_path, group, 'Features')
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            spec_set = set(self.specimens[group])

            for file in walk[2]:
                for specimen in spec_set:
                    if re.search(specimen, file) and re.search(type, file) and re.search(tissue, file):

                        df = pd.read_csv(os.path.join(f_raw_dir, file))
                        if attr in df.columns:
                            if verbose:
                                print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')
                            spec_set.remove(specimen)
                            break

            for i in spec_set:
                if verbose:
                    print(f'\t{c.FAIL}Missing complex features{c.ENDC}: {i} - {attr}')

                todo_specimens.append(i)

        return todo_specimens

    def get_features(self, spec, type='Membrane', tissue='myocardium', verbose=0, only_path=False, filtered=False):
        """
        Get features for a specific specimen.
        :param spec: Specimen to get features.
        :param type: Type of features to get. (Membrane, Nuclei)
        :param verbose: Verbosity level.
        :return: (Features path, Output path)
        """
        for group in self.specimens.keys():
            try:
                f_raw_dir = os.path.join(self.data_path, group, 'Features')
                if filtered:
                    f_raw_dir = os.path.join(self.data_path, group, 'Features', 'Filtered')
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            for file in walk[2]:
                if re.search(spec, file) and re.search(type, file) and re.search(tissue, file):
                    if filtered and not re.search('filtered', file):
                        if verbose:
                            print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')

                        path = os.path.join(f_raw_dir, file)
                        if only_path:
                            return path
                        return pd.read_csv(path)
                    else:
                        if verbose:
                            print(f'\n\t{c.OKGREEN}Found{c.ENDC}: {file}')

                        path = os.path.join(f_raw_dir, file)
                        if only_path:
                            return path
                        return pd.read_csv(path)

        raise FileNotFoundError(f'No specimen found: {spec} (Get features) [{type} - {tissue}]')

    def get_mesh_cell(self, spec, type='Membrane', tissue='myocardium', verbose=0, filtered=False):
        """
        Get cells mesh for a specific specimen.
        :param spec: Specimen to get mesh.
        :param type: Type of mesh to get. (Membrane, Nuclei)
        :param verbose: Verbosity level.
        :return: Path to the mesh file.
        """
        for group in self.specimens.keys():
            try:
                f_raw_dir = os.path.join(self.data_path, group, '3DShape', type, tissue)
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            for file in walk[2]:
                if filtered:
                    if re.search(spec, file) and re.search('filtered', file):
                        if verbose:
                            print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')
                        return os.path.join(f_raw_dir, file)
                    # else:
                    #     if re.search(spec, file) and not re.search('filtered', file):
                    #         if verbose:
                    #             print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')
                    #         return os.path.join(f_raw_dir, file)
                else:
                    if re.search(spec, file) and not re.search('filtered', file):
                        if verbose:
                            print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')
                        return os.path.join(f_raw_dir, file)

        raise FileNotFoundError(f'No specimen found: {spec} (Get mesh) [{type} - {tissue}] - Filtered: {filtered}')

    def get_mesh_tissue(self, spec, tissue='myocardium', verbose=0):
        """
        Get tissue mesh for a specific specimen.
        :param spec: Specimen to get mesh.
        :param type: Type of mesh to get. (Membrane, Nuclei)
        :param verbose: Verbosity level.
        :return: Path to the mesh file.
        """
        for group in self.specimens.keys():
            try:
                f_raw_dir = os.path.join(self.data_path, group, '3DShape/Tissue/', tissue)
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            for file in walk[2]:
                if re.search(spec, file):
                    if verbose:
                        print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')

                    return os.path.join(f_raw_dir, file)

        raise FileNotFoundError(f'No specimen found: {spec} (Get tissue mesh)]')

    def check_meshes(self, type='Membrane', tissue='myocardium', verbose=0, filtered=False):
        """
        Check if meshes have been created for each specimen.
        Get a list of not created meshes. (missing_meshes)
        :param type: Type of meshes for output path. (Membrane, Nuclei) (Default: Membrane)
        :param tissue: Tissue of meshes for output path. (Default: myocardium)
        :param verbose: Verbosity level.
        :return: List of specimens with missing meshes and their output paths.
        """
        todo_specimens = []

        for group in self.specimens.keys():
            if verbose:
                print(f'{c.OKBLUE}Group{c.ENDC}: {group}')

            try:
                f_raw_dir = os.path.join(self.data_path, group, '3DShape', type, tissue)
                walk = os.walk(f_raw_dir).__next__()
            except StopIteration:
                if verbose:
                    print(f'\t{c.FAIL}No directory{c.ENDC}: {f_raw_dir}')
                continue

            spec_set = set(self.specimens[group])

            for file in walk[2]:
                for specimen in spec_set:
                    if filtered:
                        if re.search(specimen, file) and re.search('filtered', file):
                            if verbose:
                                print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')
                            spec_set.remove(specimen)
                            break
                    else:
                        if re.search(specimen, file) and not re.search('filtered', file):
                            if verbose:
                                print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')
                            spec_set.remove(specimen)
                            break

            for i in spec_set:
                if verbose:
                    print(f'\t{c.FAIL}Missing mesh{c.ENDC}: {i}')

                todo_specimens.append(i)

        return todo_specimens

    def get_feature_map(self, spec, level='Membrane', tissue='myocardium', feature='columnarity', verbose=0):
        """
        Get feature map for a specific specimen.
        :param spec: Specimen to get feature map.
        :param level: Level of feature map. (Membrane, Nuclei)
        :param tissue: Tissue of feature map. (myocardium)
        :param feature: Feature to get. (columnarity, ...)
        :param verbose: Verbosity level.
        :return: Path to the feature map file.
        """
        group = find_group(spec)
        try:
            f_raw_dir = os.path.join(self.data_path, group, '3DShape/Tissue/', tissue, 'map', spec)
            walk = os.walk(f_raw_dir).__next__()
        except StopIteration:
            if verbose:
                raise FileNotFoundError(f'No directory: {f_raw_dir}')

        for file in walk[2]:
            if re.search(f'{level}_{feature}.ply', file):
                if verbose:
                    print(f'\t{c.OKGREEN}Found{c.ENDC}: {file}')

                return os.path.join(f_raw_dir, file)

        return None



def find_group(specimen):
    for group, specimen_list in v.specimens.items():
        if specimen in specimen_list:
            return group
    return None


def find_specimen(img_path):
    """
    Find specimen given an image path.
    :return:
    """
    for group, specimen_list in v.specimens.items():
        for specimen in specimen_list:
            if re.search(specimen, img_path):
                return specimen
    return None

