PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_INCLUDE_2=-I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

CXX = /home/dave/soft/mpi/openmpi-1.8.3/bin/mpic++

CXXFLAGS = ${MPI_INCLUDE} ${PETSC_INCLUDE} ${PETSC_INCLUDE_2} -g -Wall

OBJS = Basis.o \
       ElMats.o \
       Topo.o \
       Geom.o \
       Mimsem.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

Mimsem.o: Mimsem.cpp Geom.o Topo.o Basis.o
	${CXX} -c Mimsem.cpp ${CXXFLAGS}

Geom.o: Geom.cpp Geom.h Topo.o
	${CXX} -c Geom.cpp ${CXXFLAGS}

Topo.o: Topo.cpp Topo.h
	${CXX} -c Topo.cpp ${CXXFLAGS}

ElMats.o: ElMats.cpp ElMats.h Basis.o
	${CXX} -c ElMats.cpp ${CXXFLAGS}

Basis.o: Basis.cpp Basis.h
	${CXX} -c Basis.cpp ${CXXFLAGS}
