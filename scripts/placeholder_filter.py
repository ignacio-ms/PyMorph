import sys
import os

import numpy as np

try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.utils.timer import LoadingBar
from auxiliary.data.dataset_ht import HtDataset, find_group
from auxiliary.data import imaging


ds = HtDataset()
t = 'myocardium'

specimens = [
    "0806_E5", "0521_E1", "0119_E1", "0516_E2", "0401_E3", "0402_E2", "0209_E2", "0308_E2", "0308_E4",
    "0504_E1", "0523_E1", "0404_E1", "0503_E1", "0518_E3", "0516_E4", "0503_E2", "0401_E1", "0403_E2",
    "0521_E4", "0806_E3", "0515_E2", "0209_E1", "0521_E3", "0517_E1", "0516_E3", "0401_E2", "0404_E2",
    "0806_E4", "0516_E1", "0520_E2", "0518_E2", "0502_E1", "0516_E5",
    "0806_E6", "0518_E1", "0208_E3", "0517_E2", "0517_E4",
    "0520_E1", "0806_E1",
    "0520_E5", "0806_E2"
]
bar = LoadingBar(len(specimens) * 2)

for s in specimens:
    gr = find_group(s)
    print(f'\nSpecimen: {s} - (Group {gr})')

    for l in ['Membrane', 'Nuclei']:

        seg_path, _ = ds.read_specimen(s, l, 'Segmentation', verbose=1)
        seg = imaging.read_image(seg_path)
        try:
            split_path = seg_path.split('/')
            out_path = '/'.join(split_path[:-1]) + f'/Filtered/' + split_path[-1].replace('.nii.gz', f'_{t}.nii.gz')
            if not os.path.exists('/'.join(split_path[:-1]) + f'/Filtered/'):
                os.makedirs('/'.join(split_path[:-1]) + f'/Filtered/')
                print(f'\nCreated directory: /{" ".join(split_path[:-1])}/Filtered/')

            features = ds.get_features(s, l, t, verbose=1, filtered=True)

            cell_ids = features['cell_id']
            filtered_seg = np.zeros_like(seg, dtype=np.uint32)

            bar_2 = LoadingBar(len(cell_ids))
            for i, cell_id in enumerate(cell_ids):
                filtered_seg[seg == cell_id] = cell_id
                bar_2.update()

            bar_2.end()
            imaging.save_nii(filtered_seg, out_path, verbose=1)
        except Exception as e:
            print(f'Error in {s} {l} {t}: {e}')
        finally:
            bar.update()

bar.end()
