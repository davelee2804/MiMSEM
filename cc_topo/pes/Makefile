PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_INCLUDE_2=-I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

HDF5_INCLUDE=-I${HDF5_DIR}/include

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

CXX = mpicxx

CXXFLAGS = ${MPI_INCLUDE} ${HDF5_INCLUDE} ${PETSC_INCLUDE} ${PETSC_INCLUDE_2} -g -Wall

OBJS = LinAlg.o \
       Basis.o \
       Topo.o \
       Geom.o \
       L2Vecs.o \
       ElMats.o \
       Assembly.o \
       SWEqn.o \
       PrimEqns_HEVI.o \
       PrimEqns_HEVI2.o \
       UMJS14.o
#       Williamson2.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

#Williamson2.o: Williamson2.cpp PrimEqns_HEVI.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
#	${CXX} -c Williamson2.cpp ${CXXFLAGS}

UMJS14.o: UMJS14.cpp PrimEqns_HEVI2.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c UMJS14.cpp ${CXXFLAGS}

PrimEqns_HEVI2.o: PrimEqns_HEVI2.cpp PrimEqns_HEVI2.h SWEqn.o Assembly.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c PrimEqns_HEVI2.cpp ${CXXFLAGS}

PrimEqns_HEVI.o: PrimEqns_HEVI.cpp PrimEqns_HEVI.h SWEqn.o Assembly.o ElMats.o L2Vecs.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c PrimEqns_HEVI.cpp ${CXXFLAGS}

SWEqn.o: SWEqn.cpp SWEqn.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c SWEqn.cpp ${CXXFLAGS}

Assembly.o: Assembly.cpp Assembly.h ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Assembly.cpp ${CXXFLAGS}

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
