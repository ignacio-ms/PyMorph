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
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.data import imaging


ds = HtDataset()
bar = LoadingBar(52 * 4)

for gr in v.specimens.keys():
    print(f'Group: {gr}')
    for s in v.specimens[gr]:
        print(f'\tSpecimen: {s}')

        # lines_path, _ = ds.read_line(s, verbose=1)
        # lines = imaging.read_image(lines_path)

        for l in ['Membrane', 'Nuclei']:
            seg_path, _ = ds.read_specimen(s, l, 'Segmentation', verbose=1)
            seg = imaging.read_image(seg_path)

            for t in ['myocardium', 'splanchnic']:
                try:
                    split_path = seg_path.split('/')
                    out_path = '/'.join(split_path[:-1]) + f'/Filtered/' + split_path[-1].replace('.nii.gz', f'_{t}.nii.gz')
                    if not os.path.exists('/'.join(split_path[:-1]) + f'/Filtered/'):
                        os.makedirs('/'.join(split_path[:-1]) + f'/Filtered/')

                    if os.path.isfile(out_path):
                        bar.update()
                        continue

                    features = ds.get_features(s, l, t, verbose=1)

                    cell_ids = features['cell_id']
                    filtered_seg = np.zeros_like(seg, dtype=np.uint32)
                    for i, cell_id in enumerate(cell_ids):
                        filtered_seg[seg == cell_id] = cell_id

                    imaging.save_nii(filtered_seg, out_path, verbose=1)
                except Exception as e:
                    print(f'Error in {s} {l} {t}: {e}')
                finally:
                    bar.update()

bar.end()
