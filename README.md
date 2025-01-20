# Heart tube morphogenesis characterization

The objective is to characterize the morphogenesis of the heart tube in the early stages of development. 
The heart tube is the first structure that forms during heart development and it is the precursor of the heart.

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

## Methodology

- Nuclei segmentation
- Cell-based membrane segmentation
- Single-cell feature extraction - Radiomics features
  - single-cell feature extraction - Complex features
  - Single-cell feature extraction - Cell division
- Mesh reconstruction
- Mesh filtering
- Mesh registration
- Mesh feature mapping

Most of the implemented functions are located in the `utils` module. Method specific functions are located
in the respective module. Additionally, there are some `notebooks` that can be used as playgrounds to test the
implemented functions.

*Note: All data are located on the server and are not available in this repository. The `utils/values.py` 
file contains the path to the data, is the data is on a different location, the path must be changed.*

Organization of the project:

```
.
├── cell_division
│   ├── calibration_plots
│   ├── layers
│   └── nets
├── environments
│   └── containers
├── feature_extraction
├── filtering
├── membrane_segmentation
├── meshes
│   ├── surface_map
│   └── utils
│       ├── annotation
│       ├── features
│       └── registration
├── notebooks
├── nuclei_segmentation
│   ├── processing
│   └── quality_control
├── scripts
└── utils
    ├── data
    ├── gpu
    └── misc
```

### 1. Nuclei segmentation

Directory: `nuclei_segmentation`.

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

### 2. Cell-based membrane segmentation

### 3. Single-cell feature extraction - Radiomics features

### 3.1. Single-cell feature extraction - Complex features

### 3.2. Single-cell feature extraction - Cell division

### 4. Mesh reconstruction

### 5. Mesh filtering

### 6. Mesh registration

### 7. Mesh feature mapping



