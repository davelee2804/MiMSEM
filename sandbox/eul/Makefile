PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_INCLUDE_2=-I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

HDF5_INCLUDE=-I${HDF5_DIR}/include

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

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
       Schur.o \
       VertSolve_4.o \
       HorizSolve_4.o \
       Euler_PI_4.o \
       UMJS14_imp.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

UMJS14_imp.o: UMJS14_imp.cpp Euler_PI_4.o HorizSolve_4.o VertSolve_4.o Schur.o Assembly.o VertOps.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c UMJS14_imp.cpp ${CXXFLAGS}

Euler_PI_4.o: Euler_PI_4.cpp Euler_PI_4.h HorizSolve_4.o VertSolve_4.o Schur.o Assembly.o VertOps.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Euler_PI_4.cpp ${CXXFLAGS}

HorizSolve_4.o: HorizSolve_4.cpp HorizSolve_4.h Schur.o VertOps.o Assembly.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c HorizSolve_4.cpp ${CXXFLAGS}

VertSolve_4.o: VertSolve_4.cpp VertSolve_4.h Schur.o VertOps.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c VertSolve_4.cpp ${CXXFLAGS}

Schur.o: Schur.cpp Schur.h Assembly.o VertOps.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Schur.cpp ${CXXFLAGS}

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
