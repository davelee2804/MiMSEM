PETSC_INCLUDE=-I${PETSC_DIR}/include
PETSC_INCLUDE_2=-I${PETSC_DIR}/${PETSC_ARCH}/include
PETSC_LIB=-L${PETSC_DIR}/${PETSC_ARCH}/lib

MPI_INCLUDE=-I${MPI_DIR}/include

HDF5_INCLUDE=-I${HDF5_DIR}/include

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

CXX = mpicxx

# module load pbs openmpi/4.0.2 fftw3/3.3.8 petsc/3.12.2 hdf5/1.10.5p netcdf/4.7.3
# module load pbs openmpi/4.1.4 petsc/3.17.4 hdf5/1.10.7 netcdf/4.7.3 fftw3/3.3.8
CXXFLAGS = ${MPI_INCLUDE} ${HDF5_INCLUDE} ${PETSC_INCLUDE} ${PETSC_INCLUDE_2} -O3 -Wall
#CXXFLAGS = ${MPI_INCLUDE} ${HDF5_INCLUDE} ${PETSC_INCLUDE} ${PETSC_INCLUDE_2} -g -Wall -fsanitize=address

OBJS = LinAlg.o \
       Basis.o \
       Topo.o \
       Geom.o \
       ElMats.o \
       Assembly.o \
       ThermalSW_EEC_2.o \
       GalewskyTSW_2.o
#       SWEqn_Rosenbrock.o \
#       Williamson2.o
#       RossbyHaurwitz.o
#       SWEqn_Picard.o \
#       Galewsky.o

mimsem: ${OBJS}
	${CXX} -o mimsem ${OBJS} ${PETSC_LIB} ${CXXFLAGS}

Baroclinic.o: Baroclinic.cpp SWEqn_2L.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Baroclinic.cpp ${CXXFLAGS}

GalewskyTSW_2.o: GalewskyTSW_2.cpp ThermalSW_EEC_2.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c GalewskyTSW_2.cpp ${CXXFLAGS}

Galewsky.o: Galewsky.cpp SWEqn_Picard.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Galewsky.cpp ${CXXFLAGS}

RossbyHaurwitz.o: RossbyHaurwitz.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c RossbyHaurwitz.cpp ${CXXFLAGS}

Williamson2.o: Williamson2.cpp Test.o SWEqn.o Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c Williamson2.cpp ${CXXFLAGS}

ThermalSW.o: ThermalSW.cpp ThermalSW.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c ThermalSW.cpp ${CXXFLAGS}

ThermalSW_EEC_2.o: ThermalSW_EEC_2.cpp ThermalSW_EEC_2.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c ThermalSW_EEC_2.cpp ${CXXFLAGS}

SWEqn_Rosenbrock.o: SWEqn_Rosenbrock.cpp SWEqn_Rosenbrock.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c SWEqn_Rosenbrock.cpp ${CXXFLAGS}

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
