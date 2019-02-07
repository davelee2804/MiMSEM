PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_INCLUDE_2=-I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

HDF5_INCLUDE=-I${HDF5_DIR}/include

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CXX = mpicxx

CXXFLAGS = ${MPI_INCLUDE} ${HDF5_INCLUDE} ${PETSC_INCLUDE} ${PETSC_INCLUDE_2} -g -Wall

OBJS = LinAlg.o \
       Basis.o \
       Topo.o \
       Geom.o \
       ElMats.o \
       Assembly.o \
       AdvEqn.o \
       SWEqn_JFNK.o \
       Williamson2.o
#       Galewsky.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

Galewsky.o: Galewsky.cpp SWEqn_JFNK.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Galewsky.cpp ${CXXFLAGS}

Williamson2.o: Williamson2.cpp SWEqn_JFNK.o AdvEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Williamson2.cpp ${CXXFLAGS}

SWEqn_JFNK.o: SWEqn_JFNK.cpp SWEqn_JFNK.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c SWEqn_JFNK.cpp ${CXXFLAGS}

AdvEqn.o: AdvEqn.cpp AdvEqn.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c AdvEqn.cpp ${CXXFLAGS}

Assembly.o: Assembly.cpp Assembly.h ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Assembly.cpp ${CXXFLAGS}

ElMats.o: ElMats.cpp ElMats.h Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c ElMats.cpp ${CXXFLAGS}

Geom.o: Geom.cpp Geom.h Topo.o Basis.o
	${CXX} -c Geom.cpp ${CXXFLAGS}

Topo.o: Topo.cpp Topo.h
	${CXX} -c Topo.cpp ${CXXFLAGS}

Basis.o: Basis.cpp Basis.h
	${CXX} -c Basis.cpp ${CXXFLAGS}

LinAlg.o: LinAlg.cpp LinAlg.h
	${CXX} -c LinAlg.cpp ${CXXFLAGS}
