PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_INCLUDE_2=-I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

HDF5_INCLUDE=-I${HDF5_DIR}/include

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#
# for building on gadi:
# module load pbs openmpi/4.0.2 fftw3/3.3.8 petsc/3.12.2 hdf5/1.10.5p netcdf/4.7.3
#

CXX = mpicxx
CXXFLAGS = ${MPI_INCLUDE} ${HDF5_INCLUDE} ${PETSC_INCLUDE} ${PETSC_INCLUDE_2} -O3 -Wall

OBJS = LinAlg.o \
       Basis.o \
       Topo.o \
       Geom.o \
       L2Vecs.o \
       ElMats.o \
       VertOps.o \
       Assembly.o \
       HorizSolve.o \
       VertSolve.o \
       Euler_2.o \
       Bubble.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

Bubble.o: Bubble.cpp Euler_2.o VertSolve.o HorizSolve.o Assembly.o VertOps.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Bubble.cpp ${CXXFLAGS}

Euler_2.o: Euler_2.cpp Euler_2.h VertSolve.o HorizSolve.o Assembly.o VertOps.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Euler_2.cpp ${CXXFLAGS}

VertSolve.o: VertSolve.cpp VertSolve.h HorizSolve.o Assembly.o VertOps.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c VertSolve.cpp ${CXXFLAGS}

HorizSolve.o: HorizSolve.cpp HorizSolve.h Assembly.o VertOps.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c HorizSolve.cpp ${CXXFLAGS}

Assembly.o: Assembly.cpp Assembly.h ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Assembly.cpp ${CXXFLAGS}

VertOps.o: VertOps.cpp VertOps.h ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c VertOps.cpp ${CXXFLAGS}

ElMats.o: ElMats.cpp ElMats.h Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c ElMats.cpp ${CXXFLAGS}

L2Vecs.o: L2Vecs.cpp L2Vecs.h Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c L2Vecs.cpp ${CXXFLAGS}

Geom.o: Geom.cpp Geom.h Topo.o Basis.o
	${CXX} -c Geom.cpp ${CXXFLAGS}

Topo.o: Topo.cpp Topo.h
	${CXX} -c Topo.cpp ${CXXFLAGS}

Basis.o: Basis.cpp Basis.h
	${CXX} -c Basis.cpp ${CXXFLAGS}

LinAlg.o: LinAlg.cpp LinAlg.h
	${CXX} -c LinAlg.cpp ${CXXFLAGS}
