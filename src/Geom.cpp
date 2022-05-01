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

//#define WITH_HDF5
#define RAD_SPHERE 6371220.0
//#define RAD_SPHERE 1.0
#define COARSE 1

Geom::Geom(Topo* _topo) {
    int ii, jj;
    ifstream file;
    char filename[100];
    string line;
    double value;
    Vec vl, vg;

    topo = _topo;
    pi = topo->pi;

    sprintf(filename, "input/grid_res_quad.txt");
    file.open(filename);
    std::getline(file, line);
    quad_ord = atoi(line.c_str());
    cout << "quadrature order: " << quad_ord << endl; 
    std::getline(file, line);
    nDofsX = atoi(line.c_str());
    file.close();

    sprintf(filename, "input/local_sizes_quad_%.4u.txt", pi);
    file.open(filename);
    std::getline(file, line);
    n0l = atoi(line.c_str());
    file.close();

    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    nDofsX *= quad_ord;
    nDofs0G = n_procs*nDofsX*nDofsX + 2;

    quad = new GaussLobatto(quad_ord);
    node = new LagrangeNode(topo->elOrd, quad);
    edge = new LagrangeEdge(topo->elOrd, node);

    // determine the number of global nodes
    sprintf(filename, "input/geom_%.4u.txt", pi);
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

    sprintf(filename, "input/geom_%.4u.txt", pi);
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
        s[ii][1] = asin(x[ii][2]/RAD_SPHERE);
        ii++;
    }
    file.close();

    // topology of the quadrature points
    sprintf(filename, "input/quads_%.4u.txt", pi);
    file.open(filename);
    n0 = 0;
    while (std::getline(file, line))
        ++n0;
    file.close();

    loc0 = new int[n0];
    topo->loadObjs(filename, loc0);

    ISCreateGeneral(MPI_COMM_WORLD, n0, loc0, PETSC_COPY_VALUES, &is_g_0);
    ISCreateStride(MPI_COMM_SELF, n0, 0, 1, &is_l_0);
    VecCreateSeq(MPI_COMM_SELF, n0, &vl);
    VecCreateMPI(MPI_COMM_WORLD, n0l, nDofs0G, &vg);
    VecScatterCreate(vg, is_g_0, vl, is_l_0, &gtol_0);
    VecDestroy(&vl);
    VecDestroy(&vg);

    inds0_l = new int[(quad->n+1)*(quad->n+1)];
    inds0_g = new int[(quad->n+1)*(quad->n+1)];

    // update the global coordinates within each element for consistency with the local 
    // coordinates as defined by the Jacobian mapping
    updateGlobalCoords();

    det = new double*[topo->nElsX*topo->nElsX];
    J = new double***[topo->nElsX*topo->nElsX];
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        det[ii] = new double[(quad->n+1)*(quad->n+1)];
        J[ii] = new double**[(quad->n+1)*(quad->n+1)];
        for(jj = 0; jj < (quad->n+1)*(quad->n+1); jj++) {
            J[ii][jj] = new double*[2];
            J[ii][jj][0] = new double[2];
            J[ii][jj][1] = new double[2];
        }
    }

    initJacobians();
#ifdef COARSE
    coarseGlobalToLocal();
#endif
}

Geom::~Geom() {
    int ii, jj;

    ISDestroy(&is_l_0);
    ISDestroy(&is_g_0);
    VecScatterDestroy(&gtol_0);

    delete[] loc0;
    delete[] inds0_l;
    delete[] inds0_g;

    for(ii = 0; ii < nl; ii++) {
        delete[] x[ii];
        delete[] s[ii];
    }
    delete[] x;
    delete[] s;

    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        for(jj = 0; jj < (quad->n+1)*(quad->n+1); jj++) {
            delete[] J[ii][jj][0];
            delete[] J[ii][jj][1];
            delete[] J[ii][jj];
        }
        delete[] J[ii];
        delete[] det[ii];
    }
    delete[] J;
    delete[] det;

    delete edge;
    delete node;
    delete quad;
}

// Local to global Jacobian mapping
// Reference:
//    Guba, Taylor, Ullrich, Overfelt and Levy (2014)
//    Geosci. Model Dev. 7 2803 - 2816
void Geom::jacobian(int ex, int ey, int px, int py, double** jac) {
    int ii, jj, kk, mp1 = quad->n + 1;
    int* inds_0 = elInds0_l(ex, ey);
    double* c1 = x[inds_0[0]];
    double* c2 = x[inds_0[mp1-1]];
    double* c3 = x[inds_0[mp1*mp1-1]];
    double* c4 = x[inds_0[(mp1-1)*(mp1)]];
    double* ss = s[inds_0[py*mp1+px]];
    double x1, x2, rTildeMagInv;
    double rTilde[3];
    double A[2][3], B[3][3], C[3][4], D[4][2], AB[2][3], ABC[2][4];

    x1 = quad->x[px];
    x2 = quad->x[py];
    rTilde[0] = 0.25*((1.0-x1)*(1.0-x2)*c1[0] + (1.0+x1)*(1.0-x2)*c2[0] + (1.0+x1)*(1.0+x2)*c3[0] + (1.0-x1)*(1.0+x2)*c4[0]);
    rTilde[1] = 0.25*((1.0-x1)*(1.0-x2)*c1[1] + (1.0+x1)*(1.0-x2)*c2[1] + (1.0+x1)*(1.0+x2)*c3[1] + (1.0-x1)*(1.0+x2)*c4[1]);
    rTilde[2] = 0.25*((1.0-x1)*(1.0-x2)*c1[2] + (1.0+x1)*(1.0-x2)*c2[2] + (1.0+x1)*(1.0+x2)*c3[2] + (1.0-x1)*(1.0+x2)*c4[2]);
    rTildeMagInv = 1.0/sqrt(rTilde[0]*rTilde[0] + rTilde[1]*rTilde[1] + rTilde[2]*rTilde[2]);

    A[0][0] = -sin(ss[0]); A[0][1] = +cos(ss[0]); A[0][2] = 0.0;
    A[1][0] =         0.0; A[1][1] =         0.0; A[1][2] = 1.0;

    B[0][0] = +sin(ss[0])*sin(ss[0])*cos(ss[1])*cos(ss[1]) + sin(ss[1])*sin(ss[1]);
    B[0][1] = -0.5*sin(2.0*ss[0])*cos(ss[1])*cos(ss[1]);
    B[0][2] = -0.5*cos(ss[0])*sin(2.0*ss[1]);

    B[1][0] = -0.5*sin(2.0*ss[0])*cos(ss[1])*cos(ss[1]);
    B[1][1] = +cos(ss[0])*cos(ss[0])*cos(ss[1])*cos(ss[1]) + sin(ss[1])*sin(ss[1]);
    B[1][2] = -0.5*sin(ss[0])*sin(2.0*ss[1]);

    B[2][0] = -cos(ss[0])*sin(ss[1]);
    B[2][1] = -sin(ss[0])*sin(ss[1]);
    B[2][2] = +cos(ss[1]);

    C[0][0] = c1[0]; C[0][1] = c2[0]; C[0][2] = c3[0]; C[0][3] = c4[0];
    C[1][0] = c1[1]; C[1][1] = c2[1]; C[1][2] = c3[1]; C[1][3] = c4[1];
    C[2][0] = c1[2]; C[2][1] = c2[2]; C[2][2] = c3[2]; C[2][3] = c4[2];

    D[0][0] = -1.0 + x2; D[0][1] = -1.0 + x1;
    D[1][0] = +1.0 - x2; D[1][1] = -1.0 - x1;
    D[2][0] = +1.0 + x2; D[2][1] = +1.0 + x1;
    D[3][0] = -1.0 - x2; D[3][1] = +1.0 - x1;

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 3; jj++) {
            AB[ii][jj] = 0.0;
            for(kk = 0; kk < 3; kk++) {
                AB[ii][jj] += A[ii][kk]*B[kk][jj];
            }
        }
    }

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 4; jj++) {
            ABC[ii][jj] = 0.0;
            for(kk = 0; kk < 3; kk++) {
                ABC[ii][jj] += AB[ii][kk]*C[kk][jj];
            }
        }
    }

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 2; jj++) {
            jac[ii][jj] = 0.0;
            for(kk = 0; kk < 4; kk++) {
                jac[ii][jj] += ABC[ii][kk]*D[kk][jj];
            }
        }
    }

    jac[0][0] *= 0.25*RAD_SPHERE*rTildeMagInv;
    jac[0][1] *= 0.25*RAD_SPHERE*rTildeMagInv;
    jac[1][0] *= 0.25*RAD_SPHERE*rTildeMagInv;
    jac[1][1] *= 0.25*RAD_SPHERE*rTildeMagInv;
}

double Geom::jacDet(int ex, int ey, int px, int py, double** jac) {
    jacobian(ex, ey, px, py, jac);

    return (jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0]);
}

void Geom::interp0(int ex, int ey, int px, int py, double* vec, double* val) {
    int jj, np1, np12;
    int* inds0 = topo->elInds0_l(ex, ey);

    np1 = topo->elOrd + 1;
    np12 = np1*np1;
    //jj = py*mp1 + px;

    // assumes diagonal mass matrix for 0 forms
    //val[0] = vec[inds0[jj]];
    val[0] = 0.0;
    for(jj = 0; jj < np12; jj++) {
        val[0] += vec[inds0[jj]]*node->ljxi[px][jj%np1]*node->ljxi[py][jj/np1];
    }
}

void Geom::interp1_l(int ex, int ey, int px, int py, double* vec, double* val) {
    int jj, nn, np1, n2;
    int *inds1x, *inds1y;

    inds1x = topo->elInds1x_l(ex, ey);
    inds1y = topo->elInds1y_l(ex, ey);

    nn = topo->elOrd;
    np1 = topo->elOrd + 1;
    n2 = nn*np1;

    val[0] = 0.0;
    val[1] = 0.0;
    for(jj = 0; jj < n2; jj++) {
        val[0] += vec[inds1x[jj]]*node->ljxi[px][jj%np1]*edge->ejxi[py][jj/np1];
        val[1] += vec[inds1y[jj]]*edge->ejxi[px][jj%nn]*node->ljxi[py][jj/nn];
    }
}

void Geom::interp2_l(int ex, int ey, int px, int py, double* vec, double* val) {
    int jj, nn, n2;
    int* inds2 = topo->elInds2_l(ex, ey);

    nn = topo->elOrd;
    n2 = nn*nn;

    val[0] = 0.0;
    for(jj = 0; jj < n2; jj++) {
        val[0] += vec[inds2[jj]]*edge->ejxi[px][jj%nn]*edge->ejxi[py][jj/nn];
    }
}

void Geom::interp1_g(int ex, int ey, int px, int py, double* vec, double* val) {
    int el = ey*topo->nElsX + ex;
    int pi = py*(quad->n+1) + px;
    double val_l[2];
    double dj = det[el][pi];
    double** jac = J[el][pi];

    interp1_l(ex, ey, px, py, vec, val_l);

    val[0] = (jac[0][0]*val_l[0] + jac[0][1]*val_l[1])/dj;
    val[1] = (jac[1][0]*val_l[0] + jac[1][1]*val_l[1])/dj;
}

void Geom::interp2_g(int ex, int ey, int px, int py, double* vec, double* val) {
    int el = ey*topo->nElsX + ex;
    int pi = py*(quad->n+1) + px;
    double val_l[1];
    double dj = det[el][pi];

    interp2_l(ex, ey, px, py, vec, val_l);

    val[0] = val_l[0]/dj;
}

void Geom::write0(Vec q, char* fieldname, int tstep) {
    int ex, ey, ii, mp1, mp12;
    int* inds0;
    double val;
    char filename[100];
    PetscScalar *qArray, *qxArray;
    Vec ql, qxl, qxg;
    PetscViewer viewer;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n0, &ql);
    VecZeroEntries(ql);
    VecScatterBegin(topo->gtol_0, q, ql, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_0, q, ql, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, n0, &qxl);
    VecCreateMPI(MPI_COMM_WORLD, n0l, nDofs0G, &qxg);

    VecGetArray(ql, &qArray);
    VecGetArray(qxl, &qxArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = elInds0_l(ex, ey);
            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                interp0(ex, ey, ii%mp1, ii/mp1, qArray, &val);
                qxArray[inds0[ii]] = val;
            }
        }
    }
    VecRestoreArray(ql, &qArray);
    VecRestoreArray(qxl, &qxArray);

    VecScatterBegin(gtol_0, qxl, qxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(gtol_0, qxl, qxg, INSERT_VALUES, SCATTER_REVERSE);

#ifdef WITH_HDF5
    sprintf(filename, "output/%s_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_%.4u.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecView(qxg, viewer);
    PetscViewerDestroy(&viewer);

    VecDestroy(&ql);
    VecDestroy(&qxl);
    VecDestroy(&qxg);
}

void Geom::write1(Vec u, char* fieldname, int tstep) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    char filename[100];
    double val[2];
    Vec ul, uxg, uxl, vxl;
    PetscViewer viewer;
    PetscScalar *uArray, *uxArray, *vxArray;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecZeroEntries(ul);
    VecScatterBegin(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(topo->gtol_1, u, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecCreateSeq(MPI_COMM_SELF, n0, &uxl);
    VecCreateSeq(MPI_COMM_SELF, n0, &vxl);
    VecCreateMPI(MPI_COMM_WORLD, n0l, nDofs0G, &uxg);

    VecGetArray(ul, &uArray);
    VecGetArray(uxl, &uxArray);
    VecGetArray(vxl, &vxArray);
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = elInds0_l(ex, ey);

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                interp1_g(ex, ey, ii%mp1, ii/mp1, uArray, val);

                uxArray[inds0[ii]] = val[0];
                vxArray[inds0[ii]] = val[1];
            }
        }
    }
    VecRestoreArray(uxl, &uxArray);
    VecRestoreArray(vxl, &vxArray);
    VecRestoreArray(ul, &uArray);

    // scatter and write the zonal components
#ifdef WITH_HDF5
    sprintf(filename, "output/%s_x_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_x_%.4u.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecZeroEntries(uxg);
    VecScatterBegin(gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(gtol_0, uxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecView(uxg, viewer);
    PetscViewerDestroy(&viewer);

    // scatter and write the meridional components
#ifdef WITH_HDF5
    sprintf(filename, "output/%s_y_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_y_%.4u.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecZeroEntries(uxg);
    VecScatterBegin(gtol_0, vxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(gtol_0, vxl, uxg, INSERT_VALUES, SCATTER_REVERSE);
    VecView(uxg, viewer);
    PetscViewerDestroy(&viewer);

    VecDestroy(&ul);
    VecDestroy(&uxl);
    VecDestroy(&vxl);
    VecDestroy(&uxg);

    // also write the vector itself
    sprintf(filename, "output/%s_%.4u.vec", fieldname, tstep);
    PetscViewerBinaryOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(u, viewer);
    PetscViewerDestroy(&viewer);
}

// interpolate 2 form field to quadrature points
void Geom::write2(Vec h, char* fieldname, int tstep) {
    int ex, ey, ii, mp1, mp12;
    int *inds0;
    char filename[100];
    double val;
    Vec hxl, hxg;
    PetscScalar *hxArray, *hArray;
    PetscViewer viewer;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    VecCreateSeq(MPI_COMM_SELF, n0, &hxl);
    VecCreateMPI(MPI_COMM_WORLD, n0l, nDofs0G, &hxg);
    VecZeroEntries(hxg);

    VecGetArray(h, &hArray);
    VecGetArray(hxl, &hxArray);

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = elInds0_l(ex, ey);

            // loop over quadrature points
            for(ii = 0; ii < mp12; ii++) {
                interp2_g(ex, ey, ii%mp1, ii/mp1, hArray, &val);

                hxArray[inds0[ii]] = val;
            }
        }
    }
    VecRestoreArray(h, &hArray);
    VecRestoreArray(hxl, &hxArray);

    VecScatterBegin(gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);
    VecScatterEnd(gtol_0, hxl, hxg, INSERT_VALUES, SCATTER_REVERSE);

#ifdef WITH_HDF5
    sprintf(filename, "output/%s_%.4u.h5", fieldname, tstep);
    PetscViewerHDF5Open(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
#else
    sprintf(filename, "output/%s_%.4u.dat", fieldname, tstep);
    PetscViewerASCIIOpen(MPI_COMM_WORLD, filename, &viewer);
#endif
    VecView(hxg, viewer);

    PetscViewerDestroy(&viewer);
    VecDestroy(&hxg);
    VecDestroy(&hxl);

    // also write the vector itself
    sprintf(filename, "output/%s_%.4u.vec", fieldname, tstep);
    PetscViewerBinaryOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
    VecView(h, viewer);
    PetscViewerDestroy(&viewer);
}

// update global coordinates (cartesian and spherical) for consistency with local
// coordinates as derived from the Jacobian
void Geom::updateGlobalCoords() {
    int ex, ey, ii, jj, mp1, mp12;
    int* inds0;
    double x1, x2, rTildeMag;
    double rTilde[3];
    double *c1, *c2, *c3, *c4;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = elInds0_l(ex, ey);

            c1 = x[inds0[0]];
            c2 = x[inds0[mp1-1]];
            c3 = x[inds0[mp1*mp1-1]];
            c4 = x[inds0[(mp1-1)*(mp1)]];

            for(ii = 0; ii < mp12; ii++) {
                if(ii == 0 || ii == mp1-1 || ii == mp12-1 || ii == mp1*(mp1-1)) continue;

                jj = inds0[ii];

                x1 = quad->x[ii%mp1];
                x2 = quad->x[ii/mp1];

                rTilde[0] = 0.25*((1.0-x1)*(1.0-x2)*c1[0] + (1.0+x1)*(1.0-x2)*c2[0] + (1.0+x1)*(1.0+x2)*c3[0] + (1.0-x1)*(1.0+x2)*c4[0]);
                rTilde[1] = 0.25*((1.0-x1)*(1.0-x2)*c1[1] + (1.0+x1)*(1.0-x2)*c2[1] + (1.0+x1)*(1.0+x2)*c3[1] + (1.0-x1)*(1.0+x2)*c4[1]);
                rTilde[2] = 0.25*((1.0-x1)*(1.0-x2)*c1[2] + (1.0+x1)*(1.0-x2)*c2[2] + (1.0+x1)*(1.0+x2)*c3[2] + (1.0-x1)*(1.0+x2)*c4[2]);

                rTildeMag = sqrt(rTilde[0]*rTilde[0] + rTilde[1]*rTilde[1] + rTilde[2]*rTilde[2]);

                x[jj][0] = RAD_SPHERE*rTilde[0]/rTildeMag;
                x[jj][1] = RAD_SPHERE*rTilde[1]/rTildeMag;
                x[jj][2] = RAD_SPHERE*rTilde[2]/rTildeMag;

                s[jj][0] = atan2(x[jj][1], x[jj][0]);
                s[jj][1] = asin(x[jj][2]/RAD_SPHERE);
            }
        }
    }
}

void Geom::initJacobians() {
    int ex, ey, el, mp1, mp12, ii;

    mp1 = quad->n + 1;
    mp12 = mp1*mp1;

    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            el = ey*topo->nElsX + ex;

            for(ii = 0; ii < mp12; ii++) {
                det[el][ii] = jacDet(ex, ey, ii%mp1, ii/mp1, J[el][ii]);
            }
        }
    }
}

int* Geom::elInds0_l(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < quad->n + 1;  iy++) {
        for(ix = 0; ix < quad->n + 1; ix++) {
            inds0_l[kk] = (ey*quad->n + iy)*(nDofsX + 1) + ex*quad->n + ix;
            kk++;
        }
    }

    return inds0_l;
}

int* Geom::elInds0_g(int ex, int ey) {
    int ix, iy, kk;

    kk = 0;
    for(iy = 0; iy < quad->n + 1; iy++) {
        for(ix = 0; ix < quad->n + 1; ix++) {
            inds0_g[kk] = loc0[(ey*quad->n + iy)*(nDofsX + 1) + ex*quad->n + ix];
            kk++;
        }
    }

    return inds0_g;
}

void Geom::_Jacobian(double* c1, double* c2, double* c3, double* c4, double* xi, double* jac) {
    int ii, jj, kk;
    double rTildeMagInv, ss[2];
    double A[2][3], B[3][3], C[3][4], D[4][2], AB[2][3], ABC[2][4];

    rTildeMagInv = _LocalToGlobal(c1, c2, c3, c4, xi, ss);

    A[0][0] = -sin(ss[0]); A[0][1] = +cos(ss[0]); A[0][2] = 0.0;
    A[1][0] =         0.0; A[1][1] =         0.0; A[1][2] = 1.0;

    B[0][0] = +sin(ss[0])*sin(ss[0])*cos(ss[1])*cos(ss[1]) + sin(ss[1])*sin(ss[1]);
    B[0][1] = -0.5*sin(2.0*ss[0])*cos(ss[1])*cos(ss[1]);
    B[0][2] = -0.5*cos(ss[0])*sin(2.0*ss[1]);

    B[1][0] = -0.5*sin(2.0*ss[0])*cos(ss[1])*cos(ss[1]);
    B[1][1] = +cos(ss[0])*cos(ss[0])*cos(ss[1])*cos(ss[1]) + sin(ss[1])*sin(ss[1]);
    B[1][2] = -0.5*sin(ss[0])*sin(2.0*ss[1]);

    B[2][0] = -cos(ss[0])*sin(ss[1]);
    B[2][1] = -sin(ss[0])*sin(ss[1]);
    B[2][2] = +cos(ss[1]);

    C[0][0] = c1[0]; C[0][1] = c2[0]; C[0][2] = c3[0]; C[0][3] = c4[0];
    C[1][0] = c1[1]; C[1][1] = c2[1]; C[1][2] = c3[1]; C[1][3] = c4[1];
    C[2][0] = c1[2]; C[2][1] = c2[2]; C[2][2] = c3[2]; C[2][3] = c4[2];

    D[0][0] = -1.0 + xi[1]; D[0][1] = -1.0 + xi[0];
    D[1][0] = +1.0 - xi[1]; D[1][1] = -1.0 - xi[0];
    D[2][0] = +1.0 + xi[1]; D[2][1] = +1.0 + xi[0];
    D[3][0] = -1.0 - xi[1]; D[3][1] = +1.0 - xi[0];

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 3; jj++) {
            AB[ii][jj] = 0.0;
            for(kk = 0; kk < 3; kk++) {
                AB[ii][jj] += A[ii][kk]*B[kk][jj];
            }
        }
    }

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 4; jj++) {
            ABC[ii][jj] = 0.0;
            for(kk = 0; kk < 3; kk++) {
                ABC[ii][jj] += AB[ii][kk]*C[kk][jj];
            }
        }
    }

    for(ii = 0; ii < 2; ii++) {
        for(jj = 0; jj < 2; jj++) {
            jac[2*ii+jj] = 0.0;
            for(kk = 0; kk < 4; kk++) {
                jac[2*ii+jj] += ABC[ii][kk]*D[kk][jj];
            }
        }
    }

    jac[0] *= 0.25*RAD_SPHERE*rTildeMagInv;
    jac[1] *= 0.25*RAD_SPHERE*rTildeMagInv;
    jac[2] *= 0.25*RAD_SPHERE*rTildeMagInv;
    jac[3] *= 0.25*RAD_SPHERE*rTildeMagInv;
}

double Geom::_LocalToGlobal(double* c1, double* c2, double* c3, double* c4, double* xi, double* si) {
    double rTilde[3], rTildeMag, Xi[3];

    rTilde[0] = 0.25*((1.0-xi[0])*(1.0-xi[1])*c1[0] + (1.0+xi[0])*(1.0-xi[1])*c2[0] +
                      (1.0+xi[0])*(1.0+xi[1])*c3[0] + (1.0-xi[0])*(1.0+xi[1])*c4[0]);
    rTilde[1] = 0.25*((1.0-xi[0])*(1.0-xi[1])*c1[1] + (1.0+xi[0])*(1.0-xi[1])*c2[1] +
                      (1.0+xi[0])*(1.0+xi[1])*c3[1] + (1.0-xi[0])*(1.0+xi[1])*c4[1]);
    rTilde[2] = 0.25*((1.0-xi[0])*(1.0-xi[1])*c1[2] + (1.0+xi[0])*(1.0-xi[1])*c2[2] +
                      (1.0+xi[0])*(1.0+xi[1])*c3[2] + (1.0-xi[0])*(1.0+xi[1])*c4[2]);

    rTildeMag = sqrt(rTilde[0]*rTilde[0] + rTilde[1]*rTilde[1] + rTilde[2]*rTilde[2]);

    Xi[0] = rTilde[0]/rTildeMag;
    Xi[1] = rTilde[1]/rTildeMag;
    Xi[2] = rTilde[2]/rTildeMag;

    si[0] = atan2(Xi[1], Xi[0]);
    si[1] = asin(Xi[2]);

    return 1.0/rTildeMag;
}

bool Geom::_FindLocal(double* c1, double* c2, double* c3, double* c4, double*  theta_i, double* xi, double* jac) {
    bool found = false;
    int ii, numIts = 200;
    double eps = 1.0e-12;
    //double eps = 1.0e-14;
    //double eps = 1.0e-10;
    double theta_j[2], dTheta[2], dx[2], abs, jacInv[4], di;

    ii = 0;
    xi[0] = xi[1] = dx[0] = dx[1] = 0.0;
    do {
        _LocalToGlobal(c1, c2, c3, c4, xi, theta_j);
        dTheta[0] = theta_i[0] - theta_j[0];
        dTheta[1] = theta_i[1] - theta_j[1];
        //dTheta[0] *= eps + fabs(cos(theta_i[1]));
        dTheta[0] *= fabs(cos(theta_i[1]));

        _Jacobian(c1, c2, c3, c4, xi, jac);
        di = 1.0/(jac[0]*jac[3] - jac[1]*jac[2]);
        //
        //jac[0] /= cos(theta_i[1]);
        //jac[1] /= cos(theta_i[1]);
        //
        jacInv[0] = +di*jac[3];
        jacInv[1] = -di*jac[1];
        jacInv[2] = -di*jac[2];
        jacInv[3] = +di*jac[0];

        dx[0] = jacInv[0]*dTheta[0] + jacInv[1]*dTheta[1];
        dx[1] = jacInv[2]*dTheta[0] + jacInv[3]*dTheta[1];

        xi[0] += dx[0];
        xi[1] += dx[1];

        //abs = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
        abs = sqrt(dTheta[0]*dTheta[0] + dTheta[1]*dTheta[1]);
        ii++;
    } while(ii < numIts && abs > eps);

    if(ii < numIts && fabs(xi[0]) < 1.0 + 1.0e-6 && fabs(xi[1]) < 1.0 + 1.0e-6) found = true;
    //if(ii < numIts && fabs(xi[0]) < 1.0 + 1.0e-4 && fabs(xi[1]) < 1.0 + 1.0e-4) found = true;

    return found;
}

#define SW 0
#define SE 2
#define NE 8
#define NW 6

void Geom::coarseGlobalToLocal() {
    int ex, ey, ii, jj, proc_on_face_0, *inds0;
    double** x_dom = new double*[9];
    double** x_0 = new double*[nl];
    double** s_0 = new double*[nl];
    double c1[3], c2[3], c3[3], c4[4];
    ifstream file;
    char filename[100];
    string line;
    double value, wt_total = 0.0, wt_global, fac;
    GaussLobatto* quad = new GaussLobatto(quad_ord);

    proc_on_face_0 = topo->pi % (n_procs / 6);

    // load the global coords of the vertices (on the 0th face)
    sprintf(filename, "input/geom_%.4u.txt", proc_on_face_0);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
        x_0[ii] = new double[3];
        s_0[ii] = new double[2];
        stringstream ss(line);
        jj = 0;
        while (ss >> value) {
           x_0[ii][jj] = value;
           jj++;
        }
        s_0[ii][0] = atan2(x_0[ii][1],x_0[ii][0]);
        s_0[ii][1] = asin(x_0[ii][2]/RAD_SPHERE);
        ii++;
    }
    file.close();

    // load the global coords of the processor domain vertices (on the 0th face)
    sprintf(filename, "input/geom_coarse_%.4u.txt", proc_on_face_0);
    file.open(filename);
    ii = 0;
    while (std::getline(file, line)) {
	x_dom[ii] = new double[3];
        stringstream ss(line);
        jj = 0;
        while (ss >> value) {
           x_dom[ii][jj] = value;
           jj++;
        }
        ii++;
    }
    file.close();

    for(jj = 0; jj < 3; jj++) {
        c1[jj] = x_dom[SW][jj];
        c2[jj] = x_dom[SE][jj];
        c3[jj] = x_dom[NE][jj];
        c4[jj] = x_dom[NW][jj];
    }

    xi_coarse = new double*[nl];
    jac_coarse = new double*[nl];
    jacDet_coarse = new double[nl];
    for(ii = 0; ii < nl; ii++) {
        xi_coarse[ii] = new double[3];
        jac_coarse[ii] = new double[4];
        _FindLocal(c1, c2, c3, c4, s_0[ii], xi_coarse[ii], jac_coarse[ii]);
	jacDet_coarse[ii] = jac_coarse[ii][0]*jac_coarse[ii][3] - jac_coarse[ii][1]*jac_coarse[ii][2];
	wt_total += jacDet_coarse[ii];
    }
    MPI_Allreduce(&wt_total, &wt_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    cout << "BDDC: global weight from fine problem:    " << wt_total << endl;

    wt_coarse = new double[nl];
    for(ii = 0; ii < nl; ii++) wt_coarse[ii] = 0.0;
    for(ey = 0; ey < topo->nElsX; ey++) {
        for(ex = 0; ex < topo->nElsX; ex++) {
            inds0 = elInds0_l(ex, ey);
	    for(jj = 0; jj < (quad_ord+1)*(quad_ord+1); jj++) {
                fac = jacDet_coarse[inds0[jj]] / wt_total;
                wt_coarse[inds0[jj]] += fac*quad->w[jj%(quad_ord+1)]*quad->w[jj/(quad_ord+1)];
            }
	}
    }
    wt_total = 0.0;
    for(ii = 0; ii < nl; ii++) wt_total += wt_coarse[ii];
    MPI_Allreduce(&wt_total, &wt_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    cout << "BDDC: global weight from coarse problem:  " << wt_total << endl;

    for(ii = 0; ii < 9; ii++) {
        delete[] x_dom[ii];
    }
    delete[] x_dom;
    for(ii = 0; ii < nl; ii++) {
        delete[] x_0[ii];
        delete[] s_0[ii];
    }
    delete[] x_0;
    delete[] s_0;

    delete quad;
}
