PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

SLEPC_LIB_DIR=${SLEPC_DIR}/lib

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

include ${SLEPC_DIR}/lib/slepc/conf/slepcvariables
include ${SLEPC_DIR}/lib/slepc/conf/slepcrules

CXX = mpicxx

CXXFLAGS = ${MPI_INCLUDE} ${PETSC_INCLUDE} -O3 -Wall

OBJS = Basis.o \
       Topo.o \
       Geom.o \
       DMD.o

dmd: ${OBJS}
	${CXX} -o dmd ${OBJS} ${PETSC_LIB} ${SLEPC_LIB} ${CXXFLAGS}

DMD.o: DMD.cpp Geom.o Topo.o Basis.o
	${CXX} -c DMD.cpp ${CXXFLAGS}

Geom.o: Geom.cpp Geom.h Topo.o Basis.o
	${CXX} -c Geom.cpp ${CXXFLAGS}

Topo.o: Topo.cpp Topo.h
	${CXX} -c Topo.cpp ${CXXFLAGS}

Basis.o: Basis.cpp Basis.h
	${CXX} -c Basis.cpp ${CXXFLAGS}
