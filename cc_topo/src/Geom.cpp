#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <petscvec.h>

#include "Basis.h"
#include "Topo.h"
#include "Geom.h"

using namespace std;
using std::string;

Geom::Geom(int _pi, Topo* _topo) {
    int ii, jj;
    ifstream file;
    char filename[100];
    string line;
    double value;

    pi = _pi;
    topo = _topo;

    quad = new GaussLobatto(topo->elOrd);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // determine the number of global nodes
    sprintf(filename, "geom_%.4u.txt", pi);
    file.open(filename);
    nl = 0;
    while (std::getline(file, line))
        ++nl;
    file.close();

    if (!nl) {
        cout << "ERROR! geometry file reading: " << nl << endl;
    }

    x = new double*[nl];
    s = new double*[nl];
    for(ii = 0; ii < nl; ii++) {
        x[ii] = new double[3];
        s[ii] = new double[2];
    }

    sprintf(filename, "geom_%.4u.txt", pi);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
        stringstream ss(line);
        jj = 0;
        while (ss >> value) {
           x[ii][jj] = value;
           jj++;
        }
        s[ii][0] = atan2(x[ii][1],x[ii][0]);
        s[ii][1] = asin(x[ii][2]);
        //cout << ii << "\t" << x[ii][0] << "\t" << x[ii][1] << "\t" << x[ii][2] << endl;
        ii++;
    }
    file.close();
}

Geom::~Geom() {
    int ii;

    for(ii = 0; ii < nl; ii++) {
        delete[] x[ii];
        delete[] s[ii];
    }
    delete[] x;
    delete[] s;

    delete edge;
    delete node;
    delete quad;
}

// isoparametric jacobian, with the global coordinate approximated as an expansion over the test 
// functions. derivatives are evaluated from the lagrange polynomial derivatives within each element
void Geom::jacobian(int ex, int ey, int px, int py, double** J) {
    int ii, jj, mp1, mp12;
    int* inds_0 = topo->elInds0_l(ex, ey);
    double a, b, la, lb, dla, dlb;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;
    a = quad->x[px];
    b = quad->x[py];

    J[0][0] = J[0][1] = J[1][0] = J[1][1] = 0.0;

    for(ii = 0; ii < mp12; ii++) {
        jj = inds_0[ii];

        la = node->eval(a, ii%mp1);
        lb = node->eval(b, ii/mp1);
        dla = node->evalDeriv(a, ii%mp1);
        dlb = node->evalDeriv(b, ii/mp1);

        J[0][0] += dla*lb*s[jj][0];
        J[0][1] += la*dlb*s[jj][0];
        J[1][0] += dla*lb*s[jj][1];
        J[1][1] += la*dlb*s[jj][1];
    }
}

double Geom::jacDet(int ex, int ey, int px, int py, double** J) {
    jacobian(ex, ey, px, py, J);

    return (J[0][0]*J[1][1] - J[0][1]*J[1][0]);
}
