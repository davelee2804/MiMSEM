#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include <petscvec.h>
#include <petscviewerhdf5.h>

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

void Geom::write0(Vec q, char* fieldname, int tstep) {
    char filename[100];
    PetscViewer viewer;

    sprintf(filename, "%s_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(q, viewer);
    PetscViewerDestroy(&viewer);
}

void Geom::write1(Vec u, char* fieldname, int tstep) {
    int ex, ey, ii, jj, mp1, mp12, nn, np1, n2;
    int *inds0, *inds1x, *inds1y;
    char filename[100];
    double valx, valy, jac;
    double** J;
    Vec ul, uxg, uxl, vxl;
    PetscViewer viewer;
    PetscScalar *uArray, *uxArray, *vxArray;

    nn = topo->elOrd;
    np1 = topo->elOrd + 1;
    n2 = nn*np1;
    mp1 = topo->elOrd + 1;
    mp12 = mp1*mp1;
    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateMPI(MPI_COMM_WORLD, topo->n1, PETSC_DETERMINE, &ul);
    VecZeroEntries(ul);
    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &uxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &vxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &uxg);

    VecGetArray(ul, &uArray);
    VecGetArray(uxl, &uxArray);
    VecGetArray(vxl, &vxArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            inds1x = topo->elInds1x_l(ex, ey);
            inds1y = topo->elInds1y_l(ex, ey);

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                valx = 0.0;
                valy = 0.0;
                // loop over trial functions
                for(jj = 0; jj < n2; jj++) {
                    valx += uArray[inds1x[jj]]*node->ljxi[ii%mp1][jj%np1]*edge->ejxi[ii/mp1][jj/np1];
                    valy += uArray[inds1y[jj]]*edge->ejxi[ii%mp1][jj%nn]*node->ljxi[ii/mp1][jj/nn];
                }
                jac = jacDet(ex, ey, ii%mp1, ii/mp1, J);

                uxArray[inds0[ii]] = (J[0][0]*valx + J[0][1]*valy)/jac;
                vxArray[inds0[ii]] = (J[1][0]*valx + J[1][1]*valy)/jac;
            }
        }
    }
    VecRestoreArray(uxl, &uxArray);
    VecRestoreArray(vxl, &vxArray);
    VecRestoreArray(ul, &uArray);

    // scatter and write the zonal components
    sprintf(filename, "%s_%.4u_x.h5", fieldname, tstep);
    VecScatterBegin(topo->gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(uxg, viewer);
    PetscViewerDestroy(&viewer);

    // scatter and write the meridional components
    sprintf(filename, "%s_%.4u_y.h5", fieldname, tstep);
    VecScatterBegin(topo->gtol_0, vxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, vxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(uxg, viewer);
    PetscViewerDestroy(&viewer);

    VecDestroy(&ul);
    VecDestroy(&uxl);
    VecDestroy(&vxl);
    VecDestroy(&uxg);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}

// interpolate 2 form field to quadrature points
void Geom::write2(Vec h, char* fieldname, int tstep) {
    int ex, ey, ii, jj, mp1, mp12, nn, n2;
    int *inds0, *inds2;
    char filename[100];
    double jac, val;
    double** J;
    Vec hl, hxl, hxg;
    PetscScalar *hxArray, *hArray;
    PetscViewer viewer;

    nn = topo->elOrd;
    n2 = nn*nn;
    mp1 = topo->elOrd + 1;
    mp12 = mp1*mp1;
    J = new double*[2];
    J[0] = new double[2];
    J[1] = new double[2];

    VecCreateMPI(MPI_COMM_WORLD, topo->n2, PETSC_DETERMINE, &hl);
    VecScatterBegin(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_2, h, hl, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateMPI(MPI_COMM_WORLD, topo->n0, PETSC_DETERMINE, &hxl);
    VecCreateMPI(MPI_COMM_WORLD, topo->n0l, topo->nDofs0G, &hxg);
    VecZeroEntries(hxg);

    VecGetArray(hl, &hArray);
    VecGetArray(hxl, &hxArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = topo->elInds0_l(ex, ey);
            inds2 = topo->elInds2_l(ex, ey);

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                val = 0.0;
                // loop over trial functions
                for(jj = 0; jj < n2; jj++) {
                    val += hArray[inds2[jj]]*edge->ejxi[ii%mp1][jj%nn]*edge->ejxi[ii/mp1][jj/nn];
                }
                jac = jacDet(ex, ey, ii%mp1, ii/mp1, J);

                hxArray[inds0[ii]] = val/jac;
            }
        }
    }
    VecRestoreArray(hl, &hArray);
    VecRestoreArray(hxl, &hxArray);

    VecScatterBegin(topo->gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(topo->gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);

    sprintf(filename, "%s_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(hxg, viewer);

    PetscViewerDestroy(&viewer);
    VecDestroy(&hxg);
    VecDestroy(&hxl);
    VecDestroy(&hl);

    delete[] J[0];
    delete[] J[1];
    delete[] J;
}
