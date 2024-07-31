# Standard packages
import os
import sys
import getopt

from skimage import exposure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from csbdeep.utils import normalize
from stardist.models import StarDist3D
import tensorflow as tf

# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.gpu.gpu_tf import (
    increase_gpu_memory,
    set_gpu_allocator,
    clear_session
)
from auxiliary.data import imaging
from auxiliary.utils.colors import bcolors as c
from auxiliary.utils.timer import LoadingBar
from auxiliary.utils.bash import arg_check


def load_img(
        img_path, img_type='.nii.gz',
        normalize_img=True, equalize_img=False,
        axes='XYZ', verbose=0
):
    """
    Load and normalize image.
    :param img_path: Path to image.
    :param img_type: Extension of image[.nii.gz | .tif]. (Default: .nii.gz)
    :param normalize_img: Normalize image. (Default: True)
    :param verbose: Verbosity level.
    :return: Image.
    """
    img = imaging.read_nii(img_path, axes=axes) if img_type == '.nii.gz' else imaging.read_tiff(img_path, axes=axes)
    img = img[..., 0] if img.ndim == 4 else img

    if equalize_img:
        img = exposure.equalize_hist(img)

    if normalize_img:
        img = normalize(img, 1, 99.8, axis=(0, 1, 2))

    if verbose:
        print(f'{c.OKBLUE}Loading image{c.ENDC}: {img_path}')
        print(f'{c.BOLD}Image shape{c.ENDC}: {img.shape}')

    return img


def load_model(model_idx=1, models_path=None):
    """
    Load stardist 3D model.
    :param models_path: Path to models. (Default: '../models/stardist_models/')
    :param model_idx: Model index. (Default: 1)
        - 0: n1_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)
        - 1: n2_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)
        - 2: n3_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)
    :return: Model.
    """
    if models_path is None:
        models_path = current_dir.replace('nuclei_segmentation', 'models/stardist_models')

    model_names = [
        'n1_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)',
        'n2_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)',
        'n3_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)'
    ]

    model = StarDist3D(None, name=model_names[model_idx], basedir=models_path)
    return model


def set_gpu_strategy(gpu_strategy=None):
    """
    Set GPU strategy.
    :param gpu_strategy: GPU strategy. (Default: None)
        - None: No strategy.
        - 'Mirrored': Mirrored strategy.
        - 'MultiWorkerMirrored': MultiWorkerMirrored strategy.
    :return: GPU strategy.
    """
    if gpu_strategy is None:
        return None

    if gpu_strategy == 'Mirrored':
        strategy = tf.distribute.MirroredStrategy()
    elif gpu_strategy == 'MultiWorkerMirrored':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        raise ValueError(f'{c.FAIL}Invalid GPU strategy{c.ENDC}: {gpu_strategy} (Mirrored, MultiWorkerMirrored)')

    return strategy


def set_gpu():
    """
    Set GPU for TensorFlow.
    """
    if not tf.config.list_physical_devices('GPU'):
        print(f'{c.WARNING}No GPU available{c.ENDC}')
    else:
        print(f'{c.OKGREEN}GPU available{c.ENDC}')
        increase_gpu_memory()
        set_gpu_allocator()


def run(
    model, img,
    run_gpu=True, gpu_strategy=None,
    n_tiles=None, show_tile_progress=False,
    axes='XYZC', verbose=0
):
    """
    Run stardist 3D model on given image.
    :param model: Model.
    :param img: Image.
    :param run_gpu: Run on GPU. (Default: True)
    :param gpu_strategy: Strategy for GPU [Mirrored | MultiWorkerMirrored]. (Default: None)
    :param n_tiles:
        Out of memory (OOM) errors can occur if the input image is too large.
        To avoid this problem, the input image is broken up into (overlapping) tiles
        that are processed independently and re-assembled.
        This parameter denotes a tuple of the number of tiles for every image axis (see ``axes``).
        ``None`` denotes that no tiling should be used.
    :param show_tile_progress: Show progress during tiled predictions. (Default: False)
    :param verbose: Verbosity level.
    :return: Prediction. (Labels, Details)
    """

    def predict():
        return model.predict_instances(
            img,
            n_tiles=n_tiles, show_tile_progress=show_tile_progress,
            verbose=verbose, axes=axes
        )

    if run_gpu:
        set_gpu()
        strategy = set_gpu_strategy(gpu_strategy)
        if strategy is not None:
            with strategy.scope():
                labels, details = predict()
                clear_session()
                return labels, details

    labels, details = predict()
    clear_session()
    return labels, details


def print_usage():
    """
    Print usage of the module.
    """
    print(
        'usage: run_stardist.py -i <image> -s <specimen> -g <group> -m <model> -a <axes> -e <equalize> -c <gpu> -d <gpu_strategy> -t <n_tiles>\n'
        f'\n\n{c.BOLD}Options{c.ENDC}:'
        f'{c.BOLD}<image> [str]{c.ENDC}: Path to image\n'
        f'{c.BOLD}<specimen> [str]{c.ENDC}: Specimen to predict\n'
        f'{c.BOLD}<group> [str]{c.ENDC}: Group to predict all remaining images\n'
        '\tIf <group> is not provided, <image> is used\n'
        '\tIf not <group> nor <image> nor <specimen> are provided, all remaining images are predicted\n'
        f'{c.BOLD}<model> [int]{c.ENDC}: Model index\n'
        '\t0: n1_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)\n'
        '\t1: n2_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)\n'
        '\t2: n3_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)\n'
        f'{c.BOLD}<axes> [str]{c.ENDC}: Axes of the image\n'
        f'{c.BOLD}<equalize> [bool]{c.ENDC}: Histogram equalization over image\n'
        f'{c.BOLD}<gpu> [bool]{c.ENDC}: Run on GPU\n'
        f'{c.BOLD}<gpu_strategy> [str]{c.ENDC}: Strategy for GPU [Mirrored | MultiWorkerMirrored]\n'
        f'{c.BOLD}<n_tiles> [tuple]{c.ENDC}: Number of tiles to break up the images to be processed\n'
        f'independently and finally re-assembled\n'
        f'\tNone denotes that no tiling should be used\n'
        f'\tExample: -t "8,8,8"\n'
    )
    sys.exit(2)


if __name__ == '__main__':
    argv = sys.argv[1:]

    data_path, img, spec, group, model, axes, equalize, gpu, gpu_strategy, n_tiles, verbose = None, None, None, None, None, None, None, None, None, None, None

    try:
        opts, args = getopt.getopt(argv, 'p:i:s:g:m:a:e:c:d:t:v:', [
            'data_path=', 'img=', 'specimen=', 'group=', 'model=', 'axes=', 'equalize=', 'gpu=', 'gpu_strategy=', 'n_tiles=', 'verbose='
        ])

        if len(opts) == 0 or len(opts) > 8:
            print_usage()

        for opt, arg in opts:
            if opt in ('-p', '--data_path'):
                data_path = arg_check(opt, arg, '-p', '--img_path', str, print_usage)
            elif opt in ('-i', '--img'):
                img = arg_check(opt, arg, '-i', '--img', str, print_usage)
            elif opt in ('-s', '--specimen'):
                spec = arg_check(opt, arg, '-s', '--specimen', str, print_usage)
            elif opt in ('-g', '--group'):
                group = arg_check(opt, arg, '-g', '--group', str, print_usage)
            elif opt in ('-m', '--model'):
                model = arg_check(opt, arg, '-m', '--model', int, print_usage)
            elif opt in ('-a', '--axes'):
                axes = arg_check(opt, arg, '-a', '--axes', str, print_usage)
            elif opt in ('-e', '--equalize'):
                equalize = arg_check(opt, arg, '-e', '--equalize', bool, print_usage)
            elif opt in ('-c', '--gpu'):
                gpu = arg_check(opt, arg, '-c', '--gpu', bool, print_usage)
            elif opt in ('-d', '--gpu_strategy'):
                gpu_strategy = arg_check(opt, arg, '-d', '--gpu_strategy', str, print_usage)
            elif opt in ('-t', '--n_tiles'):
                n_tiles = arg_check(opt, arg, '-t', '--n_tiles', tuple, print_usage)
            elif opt in ('-v', '--verbose'):
                verbose = arg_check(opt, arg, '-v', '--verbose', int, print_usage)
            else:
                print(f'{c.FAIL}Invalid option{c.ENDC}: {opt}')
                print_usage()

        if data_path is None:
            data_path = v.data_path

        if model is None:
            print(f'{c.BOLD}Model not provided{c.ENDC}: Running with default model (n2_stardist_96)')
            model = 1

        if axes is None:
            print(f'{c.BOLD}Axes not specified{c.ENDC}: Set as XYZ')
            axes = 'XYZ'

        dataset = HtDataset(data_path=data_path)

        if group is not None:
            if verbose:
                print(f'Running prediction for group: {c.BOLD}{group}{c.ENDC}')

            dataset.check_segmentation(verbose=verbose)
            dataset.read_img_paths(type='RawImages')

            if group not in dataset.specimens.keys():
                print(f'{c.FAIL}Invalid group{c.ENDC}: {group}')
                print(f'{c.BOLD}Available groups{c.ENDC}: {list(dataset.specimens.keys())}')
                sys.exit(2)

            specimens = dataset.specimens[group]
            img_paths = dataset.raw_nuclei_path
            img_paths = [
                img_path for img_path in img_paths if any(
                    specimen in img_path for specimen in specimens
                )
            ]
            img_paths_out = dataset.missing_nuclei_out
            img_paths_out = [
                img_path_out for img_path_out in img_paths_out if any(
                    specimen in img_path_out for specimen in specimens
                )
            ]

        elif img is not None:
            if verbose:
                print(f'Running prediction for image: {c.BOLD}{img}{c.ENDC}')

            img_paths = [img]
            img_paths_out = [img.replace('RawImages', 'Segmentation')]

        elif spec is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on specimen{c.ENDC}: {spec}')

            img_path, img_path_out = dataset.read_specimen(spec, verbose=verbose)
            img_paths = [img_path]
            img_paths_out = [img_path_out]

        else:
            if verbose:
                print(f'Running prediction for all remaining images')

            dataset.check_segmentation(verbose=verbose)
            img_paths = dataset.missing_nuclei
            img_paths_out = dataset.missing_nuclei_out

        if len(img_paths) == 0:
            print(f'{c.WARNING}No images to predict{c.ENDC}')
            sys.exit(2)

        bar = LoadingBar(len(img_paths), length=50)
        for img_path, img_path_out in zip(img_paths, img_paths_out):
            img = load_img(img_path, equalize_img=equalize,verbose=verbose)
            model = load_model(model)

            labels, details = run(
                model, img, verbose=verbose,
                run_gpu=gpu, gpu_strategy=gpu_strategy,
                n_tiles=n_tiles, axes=axes
            )

            imaging.save_prediction(labels, img_path_out, verbose=verbose)

            bar.update()

        bar.end()

    except getopt.GetoptError:
        print_usage()
