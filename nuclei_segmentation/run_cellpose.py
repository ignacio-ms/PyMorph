# Standard libraries
import os
import sys
import getopt


# Custom packages
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(current_dir, os.pardir)))

from auxiliary import values as v
from auxiliary.utils.bash import arg_check
from auxiliary.utils.colors import bcolors as c
from auxiliary.data.dataset_ht import HtDataset
from auxiliary.utils.timer import LoadingBar

from my_cellpose import predict


def print_usage():
    """
    Print usage of script.
    """
    print(
        'usage: run_cellpose.py -i <image> -s <specimen> -gr <group> -m <model> -n <normalize> -e <equalize> -d <diameter> -c <channels> -v <verbose>'
        f'\n\n{c.BOLD}Options{c.ENDC}:\n'
        f'{c.BOLD}<image>{c.ENDC}: Path to image.\n'
        f'{c.BOLD}<specimen>{c.ENDC}: Specimen to predict.\n'
        f'\tIf <image> is not provided, <specimen> is used.\n'
        f'\tSpecimen must be in the format: XXXX_EY\n'
        f'{c.BOLD}<group>{c.ENDC}: Group to predict all remaining images.\n'
        f'\tIf <group> is not provided, <image> is used.\n'
        f'\tIn not <group> nor <image> nor <specimen> is provided, all remaining images are predicted.\n'
        f'{c.BOLD}<model>{c.ENDC}: Model to use. (Default: nuclei)\n'
        f'{c.BOLD}<normalize>{c.ENDC}: Normalize image. (Default: True)\n'
        f'{c.BOLD}<equalize>{c.ENDC}: Histogram equalization over image. (Default: True)\n'
        f'{c.BOLD}<diameter>{c.ENDC}: Diameter of nuclei. (Default: 17)\n'
        f'{c.BOLD}<channels>{c.ENDC}: Channels to use. (Default: [0, 0])\n'
        f'\tChannels must be a list of integers.\n'
        f'\t0 = Grayscale - 1 = Red - 2 = Green - 3 = Blue\n'
        f'{c.BOLD}<tissue>{c.ENDC}: Tissue to crop image. (Default: Myocardium)\n'
        f'{c.BOLD}<verbose>{c.ENDC}: Verbosity level. (Default: 0)\n'
    )
    sys.exit(2)


if __name__ == '__main__':
    argv = sys.argv[1:]

    data_path, img, spec, group, model, normalize, equalize, diameter, channels, tissue, verbose = None, None, None, None, 'nuclei', True, True, None, None, None,1

    try:
        opts, args = getopt.getopt(argv, "hp:i:s:g:m:n:e:d:c:t:v:", [
            'help', "data_path=", "image=", "specimen=", "group=", "model=", "normalize=", "equalize=", "diameter=", "channels=", "tissue=", "verbose="
        ])

        if len(opts) == 0 or len(opts) > 8:
            print_usage()

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_usage()
            elif opt in ("-p", "--data_path"):
                data_path = arg_check(opt, arg, "-p", "--data_path", str, print_usage)
            elif opt in ("-i", "--image"):
                img = arg_check(opt, arg, "-i", "--image", str, print_usage)
            elif opt in ("-s", "--specimen"):
                spec = arg_check(opt, arg, "-s", "--specimen", str, print_usage)
            elif opt in ("-g", "--group"):
                group = arg_check(opt, arg, "-g", "--group", str, print_usage)
            elif opt in ("-m", "--model"):
                model = arg_check(opt, arg, "-m", "--model", str, print_usage)
            elif opt in ("-n", "--normalize"):
                normalize = arg_check(opt, arg, "-n", "--normalize", bool, print_usage)
            elif opt in ("-e", "--equalize"):
                equalize = arg_check(opt, arg, "-e", "--equalize", bool, print_usage)
            elif opt in ("-d", "--diameter"):
                diameter = arg_check(opt, arg, "-d", "--diameter", int, print_usage)
            elif opt in ("-c", "--channels"):
                channels = arg_check(opt, arg, "-c", "--channels", list, print_usage)
            elif opt in ("-t", "--tissue"):
                tissue = arg_check(opt, arg, "-t", "--tissue", str, print_usage)
            elif opt in ("-v", "--verbose"):
                verbose = arg_check(opt, arg, "-v", "--verbose", int, print_usage)
            else:
                print(f"{c.FAIL}Invalid option{c.ENDC}: {opt}")
                print_usage()

        if data_path is None:
            data_path = v.data_path

        if normalize is None:
            normalize = True

        if equalize is None:
            equalize = True

        if model is None:
            print(f'{c.BOLD}Model not provided{c.ENDC}: Running with default model (nuclei)')
            model = 'nuclei'

        if tissue is None:
            print(f'{c.BOLD}Tissue not provided{c.ENDC}: Running with default tissues (myocardium, splanchnic)')
            # tissue = ['myocardium', 'splanchnic']
            tissue = None

        dataset = HtDataset(data_path=data_path)

        if group is not None:
            if verbose:
                print(f'{c.OKBLUE}Running prediction on group{c.ENDC}: {group}')

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
                print(f'{c.OKBLUE}Running prediction on image{c.ENDC}: {img}')

            img_paths = [data_path + img]
            img_paths_out = [img_paths[0].replace('RawImages', 'Segmentation')]
            img_paths_out = [
                img_paths_out[0].replace(
                    '_DAPI_decon_0.5',
                    f'_{model}_mask'
                )
            ]

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
            print(f'{c.WARNING}No images to process{c.ENDC}')
            sys.exit(2)

        bar = LoadingBar(len(img_paths), length=50)
        for img_path, img_path_out in zip(img_paths, img_paths_out):
            predict(
                img_path, img_path_out,
                model=model, dataset=dataset,
                normalize=normalize, equalize=equalize,
                diameter=diameter, channels=channels,
                tissue=tissue, verbose=verbose
            )

            bar.update()

        bar.end()

    except getopt.GetoptError:
        print_usage()
