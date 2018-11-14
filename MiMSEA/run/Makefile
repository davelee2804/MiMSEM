include ../Makefile

flush:
	rm ../src/*.o ../src/*.a galewsky williamson2 williamson6 keLatLon

galewsky:
	make flush; \
	cd ../src; \
		make objs; \
	cd ../run; \
	${CXX} -o galewsky Galewsky.cpp ${OBJ} \
		${PETSC_LIB} ${HDF5_LIB} ${CXXFLAGS} -L${LIB} -I${INC}

williamson2:
	make flush; \
	cd ../src; \
		make objs_w2; \
	cd ../run; \
	${CXX} -o williamson2 Williamson2.cpp ${OBJ} \
		${PETSC_LIB} ${HDF5_LIB} ${CXXFLAGS} -DW2_ALPHA -L${LIB} -I${INC}

williamson6:
	make flush; \
	cd ../src; \
		make objs; \
	cd ../run; \
	${CXX} -o williamson6 Williamson6.cpp ${OBJ} \
		${PETSC_LIB} ${HDF5_LIB} ${CXXFLAGS} -L${LIB} -I${INC}

keLatLon:
	make flush; \
	cd ../src; \
		make objs; \
	cd ../run; \
	${CXX} -o keLatLon KEAtLatLon.cpp ${OBJ} \
		${PETSC_LIB} ${HDF5_LIB} ${CXXFLAGS} -L${LIB} -I${INC}
