Heart tube morphogenesis characterization (PyMorph)

This repository contains code to characterize early morphogenesis of the mouse heart tube from 3D confocal stacks: segmentation of nuclei and cell membranes, single-cell feature extraction, 3D mesh reconstruction and registration, and feature mapping on tissue surfaces.

For context, rationale and the full scientific pipeline, see `Master_thesis_IMS.pdf` (included in this repo). The README below focuses on how to set up the environment, organize data, and run each pipeline step end-to-end on your dataset.

## Data

The dataset comprises 3D stacks of confocal microscopy images across early cardiac development. 52 embryos were imaged covering stages from early cardiac crescent to heart looping (approx. E7.75–E8.5, ~18 hours). Specimens are grouped into 10 nominal stages (1–4 crescent, 5–8 linear heart tube, 9–10 looping).

- Membranes: membrane-GFP in mesoderm (cardiogenic/cranial), membrane Tomato elsewhere
- Nuclei: DAPI
- Volumes typically span ~1 mm in XY and ~677 μm in Z (varies by specimen)
- XY resolution 0.38–0.49 μm; Z resolution 0.49–2.0 μm (stage-dependent)

Only Stage Group Representatives (SGR; medoid of each stage) were used for some analyses in the thesis due to time constraints. You can run the full pipeline on any subset (by group or by specimen).

## Repository organization

Most general-purpose code lives in `util/`. Method-specific code is under each module. Interactive demos and QA are in `notebooks/`.

```
.
├── cell_division
│   ├── calibration_plots
│   ├── layers
│   └── nets
├── environments
│   ├── containers
│   └── *.yaml  # conda environments
├── feature_extraction
├── filtering
├── membrane_segmentation
├── meshes
│   ├── surface_map
│   └── utils
│       ├── annotation
│       ├── features
│       └── registration
├── notebooks
├── nuclei_segmentation
│   ├── processing
│   └── quality_control
└── util
    ├── data
    ├── gpu
    └── misc
```

## Environment setup

Conda (mamba recommended) environments are provided in `environments/`.

1) Create the main environment (CPU/GPU-friendly):

```bash
mamba env create -f environments/py10ml.yaml
conda activate py310ml
```

2) Optional: containerized tools for surface mapping are under `environments/containers/` and `meshes/surface_map/`. See `meshes/surface_map/README.md` for the prebuilt binaries and libraries required by SurfaceMapComputation. A Singularity/Apptainer definition (`environments/containers/singularity_gpu_base.def`) and Dockerfile are included.

## Configure data path

Set your dataset root in `util/values.py`:

```
data_path = '/absolute/path/to/your/dataset/root/'
```

The repository ships with a server path that will not exist on your machine. Update it to your local or network location.

Specimen grouping and labels (e.g., `Gr1`…`Gr10`) are also defined in `util/values.py` (`specimens`, `specimens_to_analyze`, etc.). Update these if your cohort differs.

## Expected data layout

The pipeline expects the following on-disk structure under `data_path` (created incrementally by the scripts):

```
{data_path}/
  GrX/
    RawImages/
      Nuclei/
        2019{SPEC}_DAPI_decon_0.5.nii.gz
      Membrane/
        2019{SPEC}_mGFP_decon_0.5.nii.gz
    Segmentation/
      Nuclei/
        2019{SPEC}_mask.nii.gz
      Membrane/
        2019{SPEC}_mask.nii.gz
      LinesTissue/
        2019{SPEC}_lines.nii.gz              # tissue reference labels
    Features/
      2019{SPEC}_cell_properties_radiomics_{Level}_{tissue}.csv
      Filtered/
        2019{SPEC}_cell_properties_radiomics_{Level}_{tissue}.csv
    3DShape/
      Membrane/{tissue}/
        2019{SPEC}_{tissue}.ply              # cell meshes (optionally *_filtered.ply)
      Tissue/{tissue}/
        2019{SPEC}_{tissue}.ply              # tissue mesh
        cell_map/{SPEC}_cell_map.csv
        map/{SPEC}/{Level}_{feature}.ply     # colored feature map
        map/{SPEC}/{Level}_{feature}.csv     # per-face values
```

Where:

- `{SPEC}` is the specimen identifier, e.g., `0806_E3`
- `{Level}` is `Nuclei` or `Membrane`
- `{tissue}` is usually `myocardium` (see `util/values.py:lines` for available labels)

## End-to-end pipeline

All runner scripts accept `-p/--data_path` (optional) to override `util/values.data_path`. You can run by image (`-i`), by specimen (`-s`), by group (`-g`), or for all missing items (default behavior when neither `-i`, `-s`, nor `-g` are specified).

### 1) Nuclei segmentation

Option A — Cellpose:

```bash
python nuclei_segmentation/run_cellpose.py \
  -g Gr3 -m nuclei -n True -e True -d 17 -c "[0,0]" -v 1
```

Option B — StarDist 3D:

```bash
python nuclei_segmentation/run_stardist.py \
  -g Gr3 -m 0 -a XYZ -e True -c True -t "8,8,8" -v 1
```

Notes:
- Use `-s 0806_E3` to run a single specimen, or `-i relative/path/from/data_path.nii.gz` to run a single image.
- StarDist supports GPU via TensorFlow (`-c True`), and tile-based inference via `-t`.

### 2) Membrane segmentation (PlantSeg)

```bash
python membrane_segmentation/run_plantseg.py \
  -g Gr3 -t myocardium -v 1
```

If `-t/--tissue` is omitted, default tissues `["myocardium", "splanchnic"]` are used to crop/filter when relevant.

### 3) Single‑cell feature extraction — radiomics

Extract radiomics at the cell level for `Membrane` or `Nuclei` labels. Also filters connected components and can tissue-mask when requested.

```bash
python feature_extraction/run_extractor.py \
  -g Gr3 -t Membrane -l myocardium -v 1
```

Per‑specimen example (from already filtered segmentation):

```bash
python feature_extraction/run_extractor.py \
  -s 0806_E3 -t Membrane -l myocardium -v 1
```

### 3.1) Single‑cell feature extraction — complex (mesh‑based)

Compute geometric features (perpendicularity, sphericity, columnarity) on reconstructed meshes and merge into the radiomics table:

```bash
python meshes/run_extractor_complex.py \
  -g Gr3 -l Membrane -t myocardium -v 1
```

Direct paths (single case):

```bash
python meshes/run_extractor_complex.py \
  -e /abs/path/to/cells.ply \
  -p /abs/path/to/tissue.ply \
  -r /abs/path/to/features.csv \
  -o /abs/path/to/output.csv -v 1
```

### 4) Mesh reconstruction

Reconstruct meshes from segmentations and raw images:

```bash
python meshes/run_mesh_reconstruction.py \
  -g Gr3 -l Membrane -t myocardium -v 1
```

Direct paths (single case):

```bash
python meshes/run_mesh_reconstruction.py \
  -e /abs/path/to/2019{SPEC}_mask.nii.gz \
  -r /abs/path/to/2019{SPEC}_mGFP_decon_0.5.nii.gz \
  -o /abs/path/to/2019{SPEC}_myocardium.ply -v 1
```

### 5) Mesh/feature filtering (tissue intersection)

Filter features to keep only cells intersecting the tissue mesh (saves under `Features/Filtered/`):

```bash
python filtering/run_filter_tissue.py \
  -g Gr3 -l Membrane -t myocardium -v 1
```

You can also supply explicit paths (`-m/--mesh_path`, `-t/--tissue_path`, `-f/--features_path`).

### 6) Atlas‑based surface map registration

Register each specimen tissue mesh to its stage atlas and compute surface maps:

```bash
python meshes/run_surface_map.py \
  -g Gr3 -t myocardium -v 1
```

This expects atlas meshes and landmarks under:
```
{data_path}/ATLAS/{tissue}/ATLAS_{Gr}.ply
{data_path}/Landmarks/ATLAS/ATLAS_{Gr}_landmarks.pinned
{data_path}/Landmarks/2019{SPEC}_landmarks.pinned
```

See `meshes/surface_map/README.md` for SurfaceMapComputation dependencies/binaries.

### 7) Feature mapping and visualization

Map any per‑cell feature onto the tissue surface and export colored meshes/CSVs:

```bash
python meshes/run_feature_map.py \
  -g Gr3 -l Membrane -t myocardium -f columnarity -v 1
```

Outputs under `{group}/3DShape/Tissue/{tissue}/map/{SPEC}/`.

## Tips & troubleshooting

- Invalid group/specimen: check `util/values.py` (`specimens`, `specimens_to_analyze`).
- No images found: verify `data_path` and expected filenames under `RawImages/`.
- GPU unavailable (StarDist/TensorFlow): install proper CUDA drivers or set `-c False` to run on CPU with tiling.
- PlantSeg memory errors: run fewer cases, downsample, or adjust configuration inside `membrane_segmentation`.
- Atlas registration: ensure atlas/tissue meshes and `.pinned` landmark files exist with consistent group labels.

## How to cite

If you use this code, please cite the accompanying thesis (see `Master_thesis_IMS.pdf`).

## Acknowledgements

This codebase builds upon open-source tools including Cellpose, StarDist, PlantSeg, scikit-image, SimpleITK, pyRadiomics, trimesh, and others. Many thanks to all contributors and to the imaging facility for data acquisition.

