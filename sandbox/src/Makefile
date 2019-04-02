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
       ElMats.o \
       Assembly.o \
       SWEqn_Picard.o \
       SWEqn_2L.o \
       Galewsky.o
#       Baroclinic.o
#       TestDivergence.o
#       TestGradient.o
#       TestVorticity.o
#       Williamson2.o
#       RossbyHaurwitz.o
#       TestF.o
#       Test.o \

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

Baroclinic.o: Baroclinic.cpp SWEqn_2L.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Baroclinic.cpp ${CXXFLAGS}

Galewsky.o: Galewsky.cpp SWEqn_Picard.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Galewsky.cpp ${CXXFLAGS}

RossbyHaurwitz.o: RossbyHaurwitz.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c RossbyHaurwitz.cpp ${CXXFLAGS}

Williamson2.o: Williamson2.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Williamson2.cpp ${CXXFLAGS}

TestK.o: TestK.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c TestK.cpp ${CXXFLAGS}

TestF.o: TestF.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c TestF.cpp ${CXXFLAGS}

TestVorticity.o: TestVorticity.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c TestVorticity.cpp ${CXXFLAGS}

TestDivergence.o: TestDivergence.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c TestDivergence.cpp ${CXXFLAGS}

TestGradient.o: TestGradient.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c TestGradient.cpp ${CXXFLAGS}

Test.o: Test.cpp Test.h SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Test.cpp ${CXXFLAGS}

SWEqn_2L.o: SWEqn_2L.cpp SWEqn_2L.h SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c SWEqn_2L.cpp ${CXXFLAGS}

SWEqn_Picard.o: SWEqn_Picard.cpp SWEqn_Picard.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c SWEqn_Picard.cpp ${CXXFLAGS}

SWEqn.o: SWEqn.cpp SWEqn.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c SWEqn.cpp ${CXXFLAGS}

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
