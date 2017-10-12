PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_INCLUDE_2=-I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

include ${PETSC_DIR}/conf/variables
include ${PETSC_DIR}/conf/rules

CXX = /home/dave/soft/mpi/openmpi-1.8.3/bin/mpic++

CXXFLAGS = ${MPI_INCLUDE} ${PETSC_INCLUDE} ${PETSC_INCLUDE_2} -g -Wall

OBJS = Topo.o \
       Mimsem.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

Mimsem.o: Mimsem.cpp Topo.o
	${CXX} -c Mimsem.cpp ${CXXFLAGS}

Topo.o: Topo.cpp Topo.h
	${CXX} -c Topo.cpp ${CXXFLAGS}
