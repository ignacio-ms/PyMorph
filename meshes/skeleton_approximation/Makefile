# Makefile

CXX        = /usr/bin/g++ -O6 
all:	IDSet\
	PointTool\
	Skeleton

IDSet: IDSet.cxx
	$(CXX) -c IDSet.cxx
PointTool: PointTool.cxx
	$(CXX) -c PointTool.cxx
Skeleton: Skeleton.cxx
	$(CXX) -o Skeleton Skeleton.cxx IDSet.o PointTool.o  -lm

clean:
	-rm -f *.o Skeleton





