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
       Assembly.o \
       PrimEqns.o \
       JW06.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

JW06.o: JW06.cpp PrimEqns.o Assembly.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c JW06.cpp ${CXXFLAGS}

PrimEqns.o: PrimEqns.cpp PrimEqns.h Assembly.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c PrimEqns.cpp ${CXXFLAGS}

Assembly.o: Assembly.cpp Assembly.h Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Assembly.cpp ${CXXFLAGS}

Geom.o: Geom.cpp Geom.h Topo.o Basis.o
	${CXX} -c Geom.cpp ${CXXFLAGS}

Topo.o: Topo.cpp Topo.h
	${CXX} -c Topo.cpp ${CXXFLAGS}

Basis.o: Basis.cpp Basis.h
	${CXX} -c Basis.cpp ${CXXFLAGS}

LinAlg.o: LinAlg.cpp LinAlg.h
	${CXX} -c LinAlg.cpp ${CXXFLAGS}
