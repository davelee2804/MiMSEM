#include <iostream>
#include <fstream>
#include <cmath>

#include <mpi.h>
#include <petsc.h>
#include <petscvec.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"

using namespace std;

#define NX (24*3)
//#define RAD_SPHERE 6371220.0
#define RAD_SPHERE 1.0
#define LEN 1000000
#define NELS 4
#define NTHETA (2*3*4*NELS)

int rank;
bool test = true;

double LocalToGlobal(double* c1, double* c2, double* c3, double* c4, double* xi, double* si) {
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

void Jacobian(double* c1, double* c2, double* c3, double* c4, double* xi, double* jac) {
    int ii, jj, kk;
    double rTildeMagInv, ss[2];
    double A[2][3], B[3][3], C[3][4], D[4][2], AB[2][3], ABC[2][4];

    rTildeMagInv = LocalToGlobal(c1, c2, c3, c4, xi, ss);

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

bool FindLocal(double* c1, double* c2, double* c3, double* c4, double*  theta_i, double* xi) {
    bool found = false;
    int ii, numIts = 100;
    double eps = 1.0e-12;
    double theta_j[2], dTheta[2], dx[2], dxAbs, dThetaAbs, jac[4], jacInv[4], detInv;

    ii = 0;
    xi[0] = xi[1] = dx[0] = dx[1] = 0.0;
    do {
        LocalToGlobal(c1, c2, c3, c4, xi, theta_j);
        dTheta[0] = theta_i[0] - theta_j[0];
        dTheta[1] = theta_i[1] - theta_j[1];
        dTheta[0] *= eps + fabs(cos(theta_i[1]));

        Jacobian(c1, c2, c3, c4, xi, jac);
        detInv = 1.0/(jac[0]*jac[3] - jac[1]*jac[2]);
        //
        //jac[0] /= cos(theta_i[1]);
        //jac[1] /= cos(theta_i[1]);
        //
        jacInv[0] = +detInv*jac[3];
        jacInv[1] = -detInv*jac[1];
        jacInv[2] = -detInv*jac[2];
        jacInv[3] = +detInv*jac[0];

        dx[0] = jacInv[0]*dTheta[0] + jacInv[1]*dTheta[1];
        dx[1] = jacInv[2]*dTheta[0] + jacInv[3]*dTheta[1];

        xi[0] += dx[0];
        xi[1] += dx[1];

        dxAbs = sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
        dThetaAbs = sqrt(dTheta[0]*dTheta[0] + dTheta[1]*dTheta[1]);
        ii++;
    //} while(ii < numIts && dxAbs > eps);
    } while(ii < numIts && dThetaAbs > eps);

    if(ii < numIts && fabs(xi[0]) < 1.0 + 1.0e-8 && fabs(xi[1]) < 1.0 + 1.0e-8) found = true;

    return found;
}

int main(int argc, char** argv) {
    int ii, jj, kk, mp1;
    int size;
    int *inds;
    int theta_n[2], theta_m[2];
    int *els = new int[LEN];
    double *xi0 = new double[LEN];
    double *xi1 = new double[LEN];
    double *theta0 = new double[LEN];
    double *theta1 = new double[LEN];
    double dTheta;
    double theta_o[2], theta_i[2], xi[2];
    double *c1, *c2, *c3, *c4;
    static char help[] = "petsc";
    char filename[50];
    ofstream file;
    Topo* topo;
    Geom* geom;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);

    kk = 0;
    mp1 = geom->quad->n + 1;
    dTheta = 2.0*M_PI/NTHETA;
    for(ii = 0; ii < topo->nElsX*topo->nElsX; ii++) {
        // get the global coordinates of the corners of the element
        inds = topo->elInds0_l(ii%topo->nElsX, ii/topo->nElsX);
        c1 = geom->x[inds[0]];
        c2 = geom->x[inds[mp1-1]];
        c3 = geom->x[inds[mp1*mp1-1]];
        c4 = geom->x[inds[(mp1-1)*(mp1)]];

        // get the bottom left corner
        xi[0] = xi[1] = 0.0;
        LocalToGlobal(c1, c2, c3, c4, xi, theta_o);

        // get the number of increments across and up for the bottom left corner
        theta_n[0] = (theta_o[0] + 1.0*M_PI)/dTheta;
        theta_n[1] = (theta_o[1] + 0.5*M_PI)/dTheta;

        theta_n[0] = (theta_n[0] - NX/2)%NTHETA;
        theta_n[1] = (theta_n[1] - NX/2);
        if(theta_n[1] < 0) theta_n[1] = 0;

        // scan across elements and determine local coordinates for regular lon/lat
        // global coordinates
        for(jj = 0; jj < NX*NX; jj++) {
            theta_m[0] = (theta_n[0] + jj%NX)%NTHETA;
            theta_m[1] = (theta_n[1] + jj/NX);
            if(theta_m[1] > NTHETA/2 + 1) continue;

            theta_i[0] = theta_m[0]*dTheta - 1.0*M_PI;
            theta_i[1] = theta_m[1]*dTheta - 0.5*M_PI;
            theta_i[0] += 0.5*dTheta;
            if(theta_i[0] > 1.0*M_PI) theta_i[0] -= 2.0*M_PI;
            theta_i[1] += 0.5*dTheta;
            if(theta_i[1] > 0.5*M_PI) continue;

            if(FindLocal(c1, c2, c3, c4, theta_i, xi)) {
                els[kk] = ii;
                xi0[kk] = xi[0];
                xi1[kk] = xi[1];
                theta0[kk] = theta_i[0];
                theta1[kk] = theta_i[1];
                kk++;
            }
        }
    }
    cout << rank << ":\t" << kk << endl;

    sprintf(filename, "gtol_%.4u.txt", rank);
    file.open(filename);
    for(ii = 0; ii < kk; ii++) {
        file << els[ii] << "\t" << theta0[ii] << "\t" << theta1[ii] << "\t" << xi0[ii] << "\t" << xi1[ii] << endl;
    }
    file.close();

    delete topo;
    delete geom;

    delete[] els;
    delete[] xi0;
    delete[] xi1;
    delete[] theta0;
    delete[] theta1;

    PetscFinalize();

    return 0;
}
