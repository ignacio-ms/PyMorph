# Heart tube morphogenesis characterization

The objective is to characterize the morphogenesis of the heart tube in the early stages of development. 
The heart tube is the first structure that forms during heart development and it is the precursor of the heart.

1. Nuclei segmentation
2. Tissue filtering
3. Feature extraction
4. Proliferation characterization
5. Morphogenesis characterization

## Data

The data is composed of 3D stacks of confocal microscopy images of the heart tube. 
There was imaged 52 embryos in cardiac development stages ranging from the early cardica crescent until heart 
looping. The collection represents nominal ages from approximately E7.75 to E8.5 (18h).
The embryos are classified into 10 stages, 1-4 corresponding to the cardiac crescent development, 
5-8 to linear heart tube stages and 9-10 to heart looping stages. 

The mouse transgenic line of these specimens labels all the mesoderm in the head and cardiogenic area 
with membrane-GFP and cytoplasmic Tomato, while the rest of the tissues were labeled only with membrane Tomato.
All nuclei were stained using DAPI. Raw data from the three channels of all specimens, as well as some 
auxiliary files for tissue references, were provided.

The imaged volumes cover ~1mm in the X and Y dimension and a depth of ~677μm (depending on the specimen 
absolute size), which includes the whole spatial domain of the cardiogenic region and associated tissues. 
The XY pixel resolution varies from 0.38μm to 0.49μm and from 0.49μm to 2.0μm in Z, depending on the stage. 
A 1024x1024 resized version of the raw images was used for segmentation. Due to time limitations and the 
scope of this project, only the specimens identified as Stage Group Representatives (SGR) were used; that is, 
the medoid shape of each one of the 10 stages.

## Methods 

Most of the implemented functions are located in the `auxiliary` module. Method specific functions are located
in the respective module. Additionally, there are some `notebooks` that can be used as playgrounds to test the
implemented functions.

*Note: All data are located on the server and are not available in this repository. The `auxiliary/values.py` 
file contains the path to the data, is the data is on a different location, the path must be changed.*

Organization of the project:

```
.
├── auxiliary
│   ├── data
│   ├── gpu
│   └── utils
├── cell_division
├── feature_extraction
├── filtering
├── models
│   ├── cellpose_models
│   └── stardist_models
├── notebooks
└── nuclei_segmentation
```

### 1. Nuclei segmentation

The nuclei were segmented using **Cellpose**, to run the segmentation, use the following command:

```bash
python nuclei_segmentation/run_cellpose.py

usage: run_cellpose.py -i <image> -s <specimen> -gr <group> -m <model> -n <normalize> -e <equalize> -d <diameter> -c <channels> -v <verbose>

Options:
<image>: Path to image.
<specimen>: Specimen to predict.
	If <image> is not provided, <specimen> is used.
	Specimen must be in the format: XXXX_EY
<group>: Group to predict all remaining images.
	If <group> is not provided, <image> is used.
	In not <group> nor <image> nor <specimen> is provided, all remaining images are predicted.
<model>: Model to use. (Default: nuclei)
<normalize>: Normalize image. (Default: True)
<equalize>: Histogram equalization over image. (Default: True)
<diameter>: Diameter of nuclei. (Default: 17)
<channels>: Channels to use. (Default: [0, 0])
	Channels must be a list of integers.
	0 = Grayscale - 1 = Red - 2 = Green - 3 = Blue
<verbose>: Verbosity level. (Default: 0)
```

Another option is to segment the nuclei using **Stardist3D**, to run the segmentation, use the following command:

```bash

python nuclei_segmentation/run_stardist.py

usage: run_stardist.py -i <image> -s <specimen> -g <group> -m <model> -a <axes> -e <equalize> -c <gpu> -d <gpu_strategy> -t <n_tiles>

Options:
<image> [str]: Path to image
<specimen> [str]: Specimen to predict
<group> [str]: Group to predict all remaining images
	If <group> is not provided, <image> is used
	If not <group> nor <image> nor <specimen> are provided, all remaining images are predicted
<model> [int]: Model index
	0: n1_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)
	1: n2_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)
	2: n3_stardist_96_(1.6, 1, 1)_(48, 64, 64)_(1, 1, 1)
<axes> [str]: Axes of the image
<equalize> [bool]: Histogram equalization over image
<gpu> [bool]: Run on GPU
<gpu_strategy> [str]: Strategy for GPU [Mirrored | MultiWorkerMirrored]
<n_tiles> [tuple]: Number of tiles to break up the images to be processed
independently and finally re-assembled
	None denotes that no tiling should be used
	Example: -t "8,8,8"

```

### 2. Tissue filtering

In order to filter the segmented images by any tissue of interest, use the following command:

```bash

python filtering/run_filter_tissue.py

Usage: python run_filter_tissue.py -i <img> -t <tissue> -o <output> -p <data_path> -s <specimen> -g <group> -v <verbose>

Options:
<img> Input segmented image path (nii.gz or tiff).
<tissue> Tissue to filter by. (Default: myocardium)
<level> Nuclei level or Membrane level. (Default: Nuclei)
<output> Output path for filtered image. (Default: input image path with tissue name)
<data_path> Path to data directory. (Default: v.data_path)
<specimen> Specimen to filter. (Default: None)
<group> Group of specimens to filter. (Default: None)
<verbose> Verbosity level. (Default: 0)

```

### 3. Feature extraction

To extract features from the segmented images, use the following command:

```bash

python feature_extraction/run_extractor.py 

Usage: run_extractor.py -d <data_path> -s <specimen> -g <group> -t <type> -v <verbose>

Options:

<data_path>: Path to data directory.
<specimen>: Specimen to run prediction on.
<group>: Group to run prediction on.
<type>: Type of image (Nuclei, Membrane).
<tissue>: Tissue to filter by.
<verbose>: Verbosity level.

```

### 4. Proliferation characterization
