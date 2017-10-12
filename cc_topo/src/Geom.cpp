#include "Topo.h"
#include "Geom.h"

Geom::Geom(int _n0, int* inds0) {
    int ii;
    int ng;
    double *xg, *yg, *zg;

    n0 = _n0;

    x = new double*[n0];
    for(ii = 0; ii < n0; ii++) {
        x[ii] = new double[3];
    }

    // determine the number of global nodes
    sprintf(filename, "geom.txt");
    file.open(filename);
    ng = 0;
    while (std::getline(file, line))
        ++ng;
	file.close();

    xg = new double[ng];
    yg = new double[ng];
    zg = new double[ng];

    sprintf(filename, "geom.txt");
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
    }
	file.close();

	delete[] xg;
	delete[] yg;
	delete[] zg;
}

Geom::~Geom() {
    for(ii = 0; ii < n0; ii++) {
        delete[] x[ii];
    }
    delete[] x;
}
