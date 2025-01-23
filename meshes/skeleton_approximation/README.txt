Manifold Approximation of 3D Medial Axis C++ code

 Shin Yoshizawa (shin@riken.jp)

************************
This C++ codes are developed by Shin Yoshizawa at the MPII, Saarbruecken, Germany. The method is described in my paper "Free-form Skeleton-driven Mesh Deformations", Shin Yoshizawa, Alexander G. Belyaev, and Hans-Peter Seidel, ACM Solid Modeling 2003, pp. 247-253, June 16-20, 2003, University of Washington, Seattle. 

 ************************
Copyright
************************
Copyright:(c) Shin Yoshizawa, 2011.
RIKEN, Japan

 All right is reserved by Shin Yoshizawa.
This C++ code files are allowed 
for only primary user of research and educational purposes. Don't use 
secondary: copy, distribution, diversion, business purpose, and etc.. 

 In no event shall the author be liable to any party for direct, indirect, 
special, incidental, or consequential damage arising out of the use of this 
program and source files. 

************************
************************
Files
************************
The main function is in "Skeleton.cxx".

The program is constructed by the following source files:
IDList.h   Point3d.h      Polyhedron.h
IDSet.cxx   PointTool.cxx  IDSet.h    
Skeleton.cxx  PointTool.h  Point2d.h PolarList.h
************************
Compile
************************
The program is using <stdio.h>, <stdlib.h>, <unistd.h>  and <math.h>.
You can comple "make all" via Makefile.
************************
************************
Run
************************
./Skeleton PLY2input3Dmesh PLY2outout2Dmesh
************************
************************
Tips
************************
If you do not satisfy the results then I would suggest 
to apply the bitangential smoothing to the original mesh or 
 the Loop subdivision several times.
************************
Simple Manual

      How to use
          o Compile: By using attached Makefile, you may run "make". You should use the later versions of g++ 2.95.
          o Execute: Run "./Skeleton input.ply2 output.ply2". Mesh data have to be constructed by the PLY2 format.
                + Recall: you need qhull in the same directory of my program.
                + Input (input.ply2): A 2-manifold triangle mesh.
                + Output (output.ply2):A 2-manifold triangle mesh approximation of the medial axis whose vertex ID corresponds to the original vertex ID. 
      Options
      You can use the following options by changing the code in the constructor "Polyhedron()" of "Polyhedron.h". Also you should clean object file "make clean" then "make". The default setting is as follows:

      Polyhedron(){
      // Default setting
      orientation=1;
      boundarymap=1;
      BBOXCONSTANT = 1.25;
      }

          o Orientation (orientation):Integer: When you obtain the outer medial axis instead of the inner medial axis, you can change this value to 0 to extract opposite one. Because of the orientation consistency, sometime you will get the opposite one of the inner medial axis. If "orientation = 1" then the original orientation is chosen.
          o Boundary Position (boundarymap):Integer: The Voronoi diagram approximation is unstable around the original mesh boundaries because there is no points. Therefore, I set the medial axis boundaries are equal to the original mesh boundaries in default setting. You can change this value to 0 to map the medial axis boundaries to the approximated positions instead of the original mesh boundaries.
          o Bounding Box Ratio (BBOXCONSTANT):double: Sometime, the approximated Voronoi sites go far from the original mesh because of the difficulty of the Voronoi diagram calculation. This value represents a ratio of bounding box which suppress such wrong Voronoi sites. The Voronoi sites which are not inside of this bounding box are removed from candidates of medial axis vertex calculations.
      How can we get a better result ?
      The approximation quality depends on the input mesh density and the vertex distribution. To obtain a good approximation of the Voronoi sites, it is better to use a dense uniform mesh for the input mesh. If you do not satisfy the results then I would recommend to apply the bitangential smoothing or the Loop subdivision. As described in the paper, the bitangential smoothing reduces artifacts in the result caused by irregular vertex distribution or sharp features. Also it is well-known that we have to establish a dense input mesh (r-sampling) to approximate an appropriate Voronoi sites for the medial axis approximation. The Loop subdivision helps us increasing mesh density. According to my numerical experiments, the Loop subdivisions without the limit position projection is better to keep the original shape for the medial axis approximation. Although these smoothing and subdivision change the original geometry, these pre-processing are really useful to approximate the medial axis for practical applications. It is also interesting to consider the post-processing. For instance the images represented in this page are produced by this C++ code but also I applied a special smoothing (bilaplacian smoothing projected on the normal vector direction where the inner product between the normal and smoothing vector is negative if it is positive then that skeletal mesh vertex is not moved by the smoothing.) to the extracted skeletal mesh.
