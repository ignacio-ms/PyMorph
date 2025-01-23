/*
Manifold Approximation of 3D Medial Axis C++ code
Copyright:(c) Shin Yoshizawa, 2008
E-mail: shin@riken.jp

 All right is reserved by Shin Yoshizawa.
This C++ sources are allowed for only primary user of 
research and educational purposes. Don't use secondary: copy, distribution, 
diversion, business purpose, and etc.. 
 */
#define PI 3.1415926535897932385
#define next(p) p->next
#define back(p) p->back

class Polyhedron{
 public:
  int numberV;
  int numberF;
  
  Point3d **point;
  Point3d **spoint;
  Point3d **normal;
  double xmax,xmin,ymax,ymin,zmax,zmin;
  int **Face;
  int boundarymap;
  double BBOXCONSTANT;
  int orientation;
  Point3d *infinite;
  IDSet *IDtool;
  PointTool *PT;
  /*************************/
  
  // 1-ring neighbor structure
  
  IDList **IHead;
  IDList **ITail;
  IDList **VHead;
  IDList **VTail;
  IDList **FHead;
  IDList **FTail;
  int *boundary;
  int *neighborI;
  int *neighborF;
  
  int numboundary;
  Point3d **bc;
  Point3d **dbc;
    
  Polyhedron(){
    // Default setting
    orientation=1;
    boundarymap=1;
    BBOXCONSTANT = 1.25;
    
  }
  virtual ~Polyhedron(){
    if(numberV!=0&&point!=NULL){
      memorydelete();
    }
  }
  
  void readmesh(char *filename){
    
    FILE *in=NULL;
    in = fopen(filename,"r");
    int dV=0;
    int dF=0;
    int i,j;
    int di=0;
    int dj=0;
    int dk=0;
    double dx=0.0;
    double dy=0.0;
    double dz=0.0;
    fscanf(in,"%d",&dV);
    fscanf(in,"%d",&dF);
    
    memoryallocate(dV,dF);
    for(i=0;i<numberV;i++){
      fscanf(in,"%lf %lf %lf",&dx,&dy,&dz);
      setPoint(i,dx,dy,dz);
    }
    int val=3;
    for(i=0;i<numberF;i++){
      
      fscanf(in,"%d %d %d %d",&val,&di,&dj,&dk);
      if(orientation==1){
	setFace(i,di,dj,dk);
      }else{
	setFace(i,dk,dj,di);
      }
      IDtool->AppendVFSort(i,FHead[Face[i][0]],FTail[Face[i][0]]);
      IDtool->AppendVFSort(i,FHead[Face[i][1]],FTail[Face[i][1]]);
      IDtool->AppendVFSort(i,FHead[Face[i][2]],FTail[Face[i][2]]);
    }
    
    
    
    fclose(in);
    
    /* feature analysis */
    
    SetBoundaryLines();
    MakeNormals(normal);
    makeSkeleton();
   
  }
  void makeSkeleton(){
    int i,j;
    FILE *out = fopen("tempfile1.txt","w");
    fprintf(out,"%d\n",3);
    fprintf(out,"%d\n",numberV);
    for(i=0;i<numberV;i++){
      fprintf(out,"%lf %lf %lf\n",point[i]->x,point[i]->y,point[i]->z);
    }

    fclose(out);

    printf("apply qhull, please wait....\n");
    
    system("qhull < tempfile1.txt v o TO tempfile2.txt");
    
    /* This is for Windows OS. If you would use the following line then you have to comment out the above line. */
    //system("./qhull.exe < tempfile1.txt v o TO tempfile2.txt"); 
    
    printf("End qhull. Start classifications...\n");
    out = fopen("tempfile2.txt","r");
    int dim;
    int dV,dF,dD;
    Point3d **Vsites;
    IDList **VFaceHead;
    IDList **VFaceTail;
    fscanf(out,"%d",&dim);
    fscanf(out,"%d %d %d",&dV,&dF,&dD);
    //--;
    Vsites = new Point3d* [dV];
    VFaceHead = new IDList* [dF];
    VFaceTail = new IDList* [dF];
    double dx,dy,dz;
    
    
    for(i=0;i<dV;i++){
      fscanf(out,"%lf %lf %lf",&dx,&dy,&dz);
      Vsites[i] = new Point3d(dx,dy,dz);
      
    }
    // number of Voronoi cell dF should be equal to the number of input vertices
    
    int nums,dId;
    for(i=0;i<dF;i++){
      VFaceHead[i] = new IDList();
      VFaceTail[i] = new IDList();
      VFaceHead[i]->next =  VFaceTail[i];
      VFaceTail[i]->back = VFaceHead[i];
      fscanf(out,"%d",&nums);
      
      for(j=0;j<nums;j++){
	fscanf(out,"%d",&dId);
	if(dId!=0){
	  if(inside1(Vsites[dId])==1&&inside2(Vsites[dId],i)==1){
	    
	    //if(inside2(Vsites[dId],i)==1){
	  
	  IDtool->AppendVF(dId,VFaceTail[i]);
	/* if there is no Voronoi sites which satisfy these condition then
	     I assign original mesh vertex coordinates to skeleton vertex.
	     It is probably because of low resolution or cyclides.
	     I would suggest to apply the Loop subdivision for the original 
	     mesh several times, may 2times. 
	  */
	}
	}
      }
    }
    
    fclose(out);
    
    // cleaning tmp file
    out = fopen("tempfile1.txt","w");
    fprintf(out,"0\n");
    fclose(out);
    out = fopen("tempfile2.txt","w");
    fprintf(out,"0\n");
    fclose(out);
    

    for(i=0;i<dF;i++){
      if(VFaceHead[i]->next == VFaceTail[i]){
	spoint[i]->x = point[i]->x;
	spoint[i]->y = point[i]->y;
	spoint[i]->z = point[i]->z;
	
      }else{
	if(boundary[i]==1){
	  
	  if(boundarymap==1){
	    //boundary is mapped to original boundary
	    
	    spoint[i]->x = point[i]->x;
	    spoint[i]->y = point[i]->y;
	    spoint[i]->z = point[i]->z;
	    
	    
	  }else{
	    setMeanValue(i,Vsites,VFaceHead[i],VFaceTail[i]);
	    
	  }
	}else{
	  setMeanValue(i,Vsites,VFaceHead[i],VFaceTail[i]);
	}
      }
      
    }
    
    
    
    
    
      IDtool->CleanNeighborL(VFaceHead,VFaceTail,dF);
      
    
    for(i=0;i<dV;i++)delete Vsites[i];
    delete [] Vsites;
    

  }
  /* The function calculates arithmetic mean of Voronoi sites without the furthest point. The furthest point will converge to medial axis but it is the most sensitive point in the Voronoi cell. I would choice the stability instead of accuracy. If you would like to use the furthest point, you can try.

   */ 
  void setMeanValue(int ID,Point3d **vsite,IDList *vfhead,IDList *vftail){
    int i,j;
    double maxval=0.0;
    int maxID=0;
    double dval;
    IDList *now = vfhead;
    while(next(now)!=vftail){
      now = next(now);
      dval = PT->Distance(point[ID],vsite[now->ID]);
      if(dval>maxval){
	maxval = dval;
	maxID = now->ID;
      }
    }
    spoint[ID]->x = spoint[ID]->y = spoint[ID]->z= 0.0; 
    now = vfhead;
    double dnum=0.0;
    while(next(now)!=vftail){
      now = next(now);
      if(now->ID!=maxID){
	dnum++;
	spoint[ID]->x += vsite[now->ID]->x;
	spoint[ID]->y += vsite[now->ID]->y;
	spoint[ID]->z += vsite[now->ID]->z;
	
      }
    }
    if(dnum==0.0){
      spoint[ID]->x = vsite[maxID]->x;
      spoint[ID]->y = vsite[maxID]->y;
      spoint[ID]->z = vsite[maxID]->z;
    }else{
      spoint[ID]->x /= dnum;
      spoint[ID]->y /= dnum;
      spoint[ID]->z /= dnum;
    }
    
  
  }
  int inside2(Point3d *in,int jID){
    PT->makeVector(bc[0],point[jID],in);
    PT->Normalize3D(bc[0]);
    if(PT->InnerProduct(bc[0],normal[jID])<0.0){
      return 1;
    
    }
    return 0;
    
  }
  int inside1(Point3d *in){
    if(in->x >= xmin&&in->x<=xmax&&
       in->y >= ymin&&in->y<=ymax&&
       in->z >= zmin&&in->z<=zmax)return 1;
    return 0;

  }
  
  // N. Max's normal approximation
  void MakeNormals(Point3d **dNorma){
  int i,j;
  IDList *now=NULL;
  double dummytemp=0.0;
  double angle=0.0;
  double dsize1=0.0;
  double dsize2=0.0;
  double weight=0.0;
  xmax=xmin=point[0]->x;
  ymax=ymin=point[0]->y;
  zmax=zmin=point[0]->z;
  


  for(i=0;i<numberV;i++){
    if(xmax<point[i]->x)xmax = point[i]->x;
    if(ymax<point[i]->y)ymax = point[i]->y;
    if(zmax<point[i]->z)zmax = point[i]->z;
    if(xmin>point[i]->x)xmin = point[i]->x;
    if(ymin>point[i]->y)ymin = point[i]->y;
    if(zmin>point[i]->z)zmin = point[i]->z;
    

    now = VHead[i];
    bc[3]->x=0.0;
    bc[3]->y=0.0;
    bc[3]->z=0.0;
    
    while(next(now)!=VTail[i]){
      now = next(now);
      PT->makeVector(bc[0],point[i],point[next(now)->ID]);dsize1=PT->Point3dSize(bc[0]);if(dsize1==0.0)dsize1=1.0;
      PT->makeVector(bc[1],point[i],point[now->ID]);dsize2=PT->Point3dSize(bc[1]);if(dsize2==0.0)dsize2=1.0;
      
      PT->CrossVector(bc[2],bc[1],bc[0]);
      weight = 1.0/(dsize1*dsize1*dsize2*dsize2);
      bc[3]->x += weight*bc[2]->x;
      bc[3]->y += weight*bc[2]->y;
      bc[3]->z += weight*bc[2]->z;
      

      now = next(now);
    }
    dummytemp = PT->Point3dSize(bc[3]);
    if(dummytemp != 0.0){
      dNorma[i]->x = ((bc[3]->x)/dummytemp);
      dNorma[i]->y = ((bc[3]->y)/dummytemp);
      dNorma[i]->z = ((bc[3]->z)/dummytemp);
    }
    
  }
  
  double xcenter = 0.5*(xmax+xmin);
  xmax = BBOXCONSTANT*(xmax-xcenter)+xcenter;
  xmin = BBOXCONSTANT*(xmin-xcenter)+xcenter;
  double ycenter = 0.5*(ymax+ymin);
  ymax = BBOXCONSTANT*(ymax-ycenter)+ycenter;
  ymin = BBOXCONSTANT*(ymin-ycenter)+ycenter;
  double zcenter = 0.5*(zmax+zmin);
  zmax = BBOXCONSTANT*(zmax-zcenter)+zcenter;
  zmin = BBOXCONSTANT*(zmin-zcenter)+zcenter;
  

  printf("%lf %lf\n",xmin,xmax);
  printf("%lf %lf\n",ymin,ymax);
  printf("%lf %lf\n",zmin,zmax);
  
  
}


  void writemesh(char *filename){
    int i=0;
    FILE *out = fopen(filename,"w");
    fprintf(out,"%d\n",numberV);
    fprintf(out,"%d\n",numberF);
    for(i=0;i<numberV;i++){
      fprintf(out,"%lf %lf %lf\n",spoint[i]->x,spoint[i]->y,spoint[i]->z);
    }
    for(i=0;i<numberF;i++)
      fprintf(out,"3 %d %d %d\n",Face[i][0],Face[i][1],Face[i][2]);
    fclose(out);
    
  }
  

  Polyhedron(const Polyhedron& rhs);
  const Polyhedron &operator=(const Polyhedron& rhs);




private:


  
  void memorydelete(){
    int i;
    for(i=0;i<10;i++){
      delete bc[i];
      delete dbc[i];
    }
    delete infinite;
    delete [] bc;
    bc = NULL;
    delete [] dbc;
    dbc = NULL;
    
    if(IDtool!=NULL){
      if(FHead!=NULL){
	IDtool->CleanNeighborL(FHead,FTail,numberV);
	
      }
      if(IHead!=NULL&&ITail!=NULL){
	IDtool->CleanNeighborL(IHead,ITail,numberV);
	IHead=NULL;
	ITail=NULL;
      }
          
      if(VHead!=NULL&&VTail!=NULL){
	IDtool->CleanNeighborL(VHead,VTail,numberV);
	VHead=NULL;
	VTail=NULL;
      }
      delete IDtool;
      IDtool=NULL; 
    }
    
  if(point!=NULL){
    if(numberV!=0){
      for(i=0;i<numberV;i++){
	delete point[i];
	delete spoint[i];
	delete normal[i];
      }
    }
    delete [] normal;
    delete [] spoint;
    delete [] point;
    
  }
  if(Face!=NULL){
    if(numberF!=0){
      for(i=0;i<numberF;i++)delete [] Face[i];
    }
    delete [] Face;
    Face=NULL;
  }
  if(neighborI!=NULL){
    delete [] neighborI;
  }
 if(neighborF!=NULL){
    delete [] neighborF;
  }
 if(boundary!=NULL){
    delete [] boundary;
  }
 delete PT;
  }  
  
  void memoryallocate(int dV,int dF){
    if(numberV!=0){
      memorydelete();
    }
    infinite = new Point3d();  
    numberV=dV;
    numberF=dF;
    point = new Point3d* [numberV];
    spoint = new Point3d* [numberV];
    normal = new Point3d* [numberV];
    Face = new int* [numberF];
    IDtool = new IDSet();
    IHead = new IDList* [numberV];
    ITail = new IDList* [numberV];
    
  
    VHead = new IDList* [numberV];
    VTail = new IDList* [numberV];
    FHead= new IDList* [numberV];
    FTail= new IDList* [numberV];
    neighborF= new int[numberV];
    boundary = new int[numberV];
    neighborI = new int[numberV]; 
        
    numboundary=0;
    bc = new Point3d* [10];
    dbc = new Point3d* [10];
    int i;
    for(i=0;i<10;i++){
      bc[i] = new Point3d(0.0,0.0,0.0);
      dbc[i] = new Point3d(0.0,0.0,0.0);
    }
  }
  void setPoint(int i,double dx,double dy,double dz){
  point[i] = new Point3d(dx,dy,dz);
  spoint[i] = new Point3d(0.0,0.0,0.0);
  normal[i] = new Point3d(0.0,0.0,0.0);
  IHead[i] = new IDList();
  ITail[i] = new IDList();
  IHead[i]->next = ITail[i];
  ITail[i]->back = IHead[i];
  FHead[i] = new IDList();
  FTail[i] = new IDList();
  FHead[i]->next = FTail[i];
  FTail[i]->back = FHead[i];
  
  VHead[i] = new IDList();
  VTail[i] = new IDList();
  VHead[i]->next = VTail[i];
  VTail[i]->back = VHead[i];
  
}
  void SetBoundaryLines(){
    int i=0;numboundary=0;
    if(neighborI!=NULL && neighborF!=NULL && boundary!=NULL)
      for(i=0;i<numberV;i++){
	if(((neighborI[i]) == neighborF[i]) && 
	   (neighborF[i] !=0) && 
	   (neighborI[i] !=0)){
	  boundary[i] = 0;
	}else{
	  boundary[i] = 1;
	  numboundary++;
	}
      }
    //printf("number of boundary points = %d\n",numboundary);
  }
  void setFace(int i,int di,int dj,int dk){
  Face[i] = new int[3];
  Face[i][0] = di;
  Face[i][1] = dj;
  Face[i][2] = dk;
  /* One */
  neighborF[di]++;
  
  IDtool->AppendISort(dj,IHead[di],ITail[di],di,neighborI);
  IDtool->AppendISort(dk,IHead[di],ITail[di],di,neighborI);
  IDtool->AppendVF(dj,VTail[di]);
  IDtool->AppendVF(dk,VTail[di]);
  /* Two */
  neighborF[dj]++;
  
  IDtool->AppendISort(di,IHead[dj],ITail[dj],dj,neighborI);
  IDtool->AppendISort(dk,IHead[dj],ITail[dj],dj,neighborI);
  IDtool->AppendVF(dk,VTail[dj]);
  IDtool->AppendVF(di,VTail[dj]);
  /* Three */
  neighborF[dk]++;
  
  IDtool->AppendISort(di,IHead[dk],ITail[dk],dk,neighborI);
  IDtool->AppendISort(dj,IHead[dk],ITail[dk],dk,neighborI);
  IDtool->AppendVF(di,VTail[dk]);
  IDtool->AppendVF(dj,VTail[dk]);
  }

  
  

};
