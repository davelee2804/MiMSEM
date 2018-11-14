include ../Makefile

objs: LinAlg.o \
      Basis.o \
      Topo.o \
      Geom.o \
      ElMats.o \
      Assembly.o \
      SWEqn.o \
      libsrc.a

objs_w2: LinAlg.o \
      Basis.o \
      Topo.o \
      Geom.o \
      ElMats.o \
      Assembly.o \
      SWEqn_w2.o \
      libsrc.a

SWEqn_w2.o: SWEqn.cpp SWEqn.h Assembly.o ElMats.o Geom.o Topo.o Basis.o LinAlg.o
	${CXX} -c SWEqn.cpp ${CXXFLAGS} -DW2_ALPHA

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

libsrc.a:
	ar -cq libsrc.a *.o
#    mv libbase.a ../lib
#    cp *.h ../inc
#    mv *.o ../obj
