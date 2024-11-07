README for program SurfaceMapComputation and ViewMap

Software linked to publication: https://www.biorxiv.org/content/10.1101/2021.10.07.463475v1

LICENSE OF USE
3-Clause BSD License

VERSION 1.0

INTRODUCTION
The SurfaceMapComputation software computes the surface map between any pair of 0-genus closed surfaces. Additionally, the software performs a rigid registration of the shapes (this facilitates the visual inspection of both shapes/meshes simultaneously). External dependencies, instructions and a test case are provided.
The visualisation tool (ViewMap) is provided to allow inspection of the resulting mapping: manual adjustment of the landmarks, adding new landmakrs and/or trying different initialization algorithms, which is key for finding the right mapping between two shapes. 
The software takes as input parameters two shapes, their landmarks (minimum 4), number of iterations and the algorithm that is used for the initialization of the map. The appearance of swirls (paths between landmarks that are mapped to the target surface and take excessively long routes) is influenced by the initialization algorithm and by the landmarks. The initialization algorithm can be set to be decided automatically by the software, or for manual selection by the user from three different algorithms (#1 Schreiner 2004 et al., #2 Born 2021 et al., #3 Born 2021 et al.). 
Once the surface map is been computed, the application “ViewMap” can be utilized to visualize the resulting file “latest.map”. ViewMap will open windows displaying the mesh of shapeA and the mesh of shapeA mapped to shapeB. Both windows can be controlled at once by using the mouse. Paths can be selected from the vertices in one mesh (using the mouse scroll function) and the equivalent path will be automatically displayed on the second shape.
A test case is provided in which the surface map between the myocardium shapes of E13 (shapeA.ply) and E11 (shapeB.ply). It includes the files with the number of the vertices representing the landmarks (lm1; lm2; lm3; lm4; lm11; lm12 two landmarks located between lm1 and lm12, and lm4 and lm12, placed along curve 2; and two landmarks located between lm1 and lm3, and lm2 and lm4, placed along curves 3; see Fig. 12c). The output of this test has also been included and has been computed by setting the initialization algorithm in automatic selection and two different numbers of pre-iterations; 100 and 500.

PRE-REQUISITES
Operating system: this software only runs on Linux (tested on Ubuntu 18.04.6, Ubuntu 20.04 and Rocky Linux 8.4)
OpenGL version must be at least 4.5. You can check this by running $ glxinfo | grep OpenGL
If not already present:
    · OpenGL (e.g. $ sudo apt-get install libgl1-mesa-dev mesa-utils)

This program is a compiled standalone software which does need linking against external shared libraries (included in the .zip file):
1. libs/libglow.so
2. libs/libglow-extras
3. libs/libOpenMeshCore.so.9.0
4. libs/libglad.so
Include these in your system (e.g.: /path/to/sharedlibraries) and add them to your LD_LIBRARY_PATH variable ($ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/sharedlibraries)

INSTALLATION
1. Include libraries in the LD_LIBRARY_PATH variable (see above)
2. cd /SurfaceMapComputation
3. Run the executable to compute a surface map: $ ./SurfaceMapComputation
4. Run the executable to visualize the resulting map: $ ./ViewMap
NOTE: we provide independent compiled version for running either in Intel or AMD processors. Run the executable from the folders (/INTEL or /AMD) that are suited for your system.

USAGE
A typical usage for this program is:
$ ./SurfaceMapComputation --path </path/to/test_SurfaceMapComputation> --init_method 0 --shapeA <origin_shape>.ply --shapeB <target_shape>.ply --landmarksA <origin_shape>.pinned --landmarksB <target_shape>.pinned --n_pre_iters 100 --n_main_iters 500

    where:
     --path: absolute path to the folder containing all the necessary data (origin and target shape and landmarks)
     --init_method: initialization method of the algorithm: 0 (automatic), 1 (Schreiner), 2 (Born A -> B), 3 (Born B -> A)
     --shapeA: name of the origin shape, located withtin <path> in .ply format
     --shapeB: name of the target shape, located within <path> in .ply format
     --landmarksA: name of the landmarks file corresponding to <shapeA> in .pinned format, default: name of <shapeA>
     --landmarksB: name of the landmarks file corresponding to <shapeA> in .pinned format, default: name of <shapeB>
     --n_pre_iters: number of pre-iterations, default: 100
     --n_main_iters: number of iterations, default: 500

Example:
$./SurfaceMapComputation --path /path/to/test_SurfaceMapComputation --init_method 0 --shapeA shapeA.ply --shapeB shapeB.ply --landmarksA shapeA.pinned --landmarksB shapeB.pinned --n_pre_iters 100 --n_main_iters 500

The approximate execution time for the test case included in the folder <test_SurfaceMapComputation> is of:
    -2008.5  [sec] (i9-10900 2.80GHz)
    -3812.44 [sec] (Xeon E5-1620 v3 3.50GHz)
    -4652.05 [sec] (i7-4710HQ 2.50GHz)
The outcome of the program is included in the folder /test_SurfaceMapComputation_output

In order to visualize the resulting map:
$ ./ViewMap --path </path/to/test_SurfaceMapComputation/map/latest.map>

REFERENCES
[1] Schmidt, P., Campen, M., Born, J. & Kobbelt, L. Inter-surface maps via constant-curvature metrics. ACM Trans. Graph. 39, Article 119, doi:10.1145/3386569.3392399 (2020)
[2] Born, J., Schmidt, P. & Kobbelt, L. Layout Embedding via Combinatorial Optimization. Computer Graphics Forum 40, 277-290, doi:https://doi.org/10.1111/cgf.142632 (2021)