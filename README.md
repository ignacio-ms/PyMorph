# Heart tube morphogenesis characterization

The objective is to characterize the morphogenesis of the heart tube in the early stages of development. 
The heart tube is the first structure that forms during heart development and it is the precursor of the heart.

- Nuclei segmentation
- Feature extraction
- Proliferation characterization
- Morphogenesis characterization

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

### 1. Nuclei segmentation


