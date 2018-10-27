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
#include "ElMats.h"
#include "Assembly.h"
#include "SWEqn.h"

using namespace std;

#define NX (24*3)
//#define RAD_SPHERE 6371220.0
#define RAD_SPHERE 1.0
#define LEN 1000000
//#define NELS 4
#define NELS 32
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
    int ii, numIts = 200;
    double eps = 1.0e-12;
    //double eps = 1.0e-14;
    //double eps = 1.0e-10;
    double theta_j[2], dTheta[2], dx[2], abs, jac[4], jacInv[4], di;

    ii = 0;
    xi[0] = xi[1] = dx[0] = dx[1] = 0.0;
    do {
        LocalToGlobal(c1, c2, c3, c4, xi, theta_j);
        dTheta[0] = theta_i[0] - theta_j[0];
        dTheta[1] = theta_i[1] - theta_j[1];
        //dTheta[0] *= eps + fabs(cos(theta_i[1]));
        dTheta[0] *= fabs(cos(theta_i[1]));

        Jacobian(c1, c2, c3, c4, xi, jac);
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

double Interp2(Geom* geom, PetscScalar* kArray, double* c1, double* c2, double* c3, double* c4, int el, double* xi) {
    int ii;
    int n = geom->topo->elOrd;
    int ex = el%geom->topo->nElsX;
    int ey = el/geom->topo->nElsX;
    int* inds = geom->topo->elInds2_l(ex, ey);
    double ei, ej;
    double u2 = 0.0;
    double jac[4], det;
    LagrangeEdge* edge = geom->edge;

    Jacobian(c1, c2, c3, c4, xi, jac);
    det = jac[0]*jac[3] - jac[1]*jac[2];

    for(ii = 0; ii < n*n; ii++) {
        ei = edge->eval(xi[0], ii%n);
        ej = edge->eval(xi[1], ii/n);
        u2 += kArray[inds[ii]]*ei*ej;
    }
    u2 /= det*6371220.0*6371220.0;

    return u2;
}

void KineticEnergy(SWEqn* sw, Vec ui, Vec ke) {
    Vec ul, u2;
    KSP ksp;

    if(!rank) cout << "solving for kinetic energy..." << endl;

    VecCreateSeq(MPI_COMM_SELF, sw->topo->n1, &ul);
    VecCreateMPI(MPI_COMM_WORLD, sw->topo->n2l, sw->topo->nDofs2G, &u2);

    VecScatterBegin(sw->topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sw->topo->gtol_1, ui, ul, INSERT_VALUES, SCATTER_FORWARD);

    VecZeroEntries(ke);
    sw->K->assemble(ul);
    MatMult(sw->K->M, ui, u2);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, sw->M2->M, sw->M2->M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOptionsPrefix(ksp,"test_ke_");
    KSPSetFromOptions(ksp);
    KSPSolve(ksp, u2, ke);

    if(!rank) cout << "\t...done." << endl;

    KSPDestroy(&ksp);
    VecDestroy(&ul);
    VecDestroy(&u2);
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
    double *kTheta = new double[LEN];
    double dTheta;
    double theta_o[2], theta_i[2], xi[2];
    double *c1, *c2, *c3, *c4;
    static char help[] = "petsc";
    char filename[50];
    ofstream file;
    PetscScalar* kArray;
    Vec ui, ke, kel;
    PetscViewer viewer;
    Topo* topo;
    Geom* geom;
    SWEqn* sw;

    PetscInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cout << "importing topology for processor: " << rank << " of " << size << endl;

    topo = new Topo(rank);
    geom = new Geom(rank, topo);
    sw = new SWEqn(topo, geom);

    VecCreateSeq(MPI_COMM_SELF, sw->topo->n2, &kel);
    VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ui);
    VecCreateMPI(MPI_COMM_WORLD, topo->n2l, topo->nDofs2G, &ke);

    sprintf(filename, "output_gal_32x3p_dt40s/velocity_0168.vec");
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
    VecLoad(ui, viewer);
    PetscViewerDestroy(&viewer);

    KineticEnergy(sw, ui, ke);
    VecScatterBegin(sw->topo->gtol_2, ke, kel, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(sw->topo->gtol_2, ke, kel, INSERT_VALUES, SCATTER_FORWARD);
    VecGetArray(kel, &kArray);

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
        theta_n[0] = (int)((theta_o[0] + 1.0*M_PI)/dTheta);
        theta_n[1] = (int)((theta_o[1] + 0.5*M_PI)/dTheta);

        theta_n[0] = (theta_n[0] - NX/2)%NTHETA;
        theta_n[1] = (theta_n[1] - NX/2);
        if(theta_n[1] < 0) theta_n[1] = 0;

        // scan across elements and determine local coordinates for regular lon/lat
        // global coordinates
        for(jj = 0; jj < NX*NX; jj++) {
            theta_m[0] = (theta_n[0] + jj%NX)%NTHETA;
            theta_m[1] = (theta_n[1] + jj/NX);
            if(theta_m[1] > NTHETA/2 + 1) continue;

            theta_i[0] = theta_m[0]*dTheta - 1.0*M_PI + 0.5*dTheta;
            theta_i[1] = theta_m[1]*dTheta - 0.5*M_PI + 0.5*dTheta;
            if(theta_i[0] > 1.0*M_PI) theta_i[0] -= 2.0*M_PI;
            if(theta_i[1] > 0.5*M_PI) continue;

            if(FindLocal(c1, c2, c3, c4, theta_i, xi)) {
                els[kk] = ii;
                xi0[kk] = xi[0];
                xi1[kk] = xi[1];
                theta0[kk] = theta_i[0];
                theta1[kk] = theta_i[1];
                kTheta[kk] = Interp2(geom, kArray, c1, c2, c3, c4, ii, xi);

                kk++;
            }
        }
    }
    VecRestoreArray(kel, &kArray);
    cout << rank << ":\t" << kk << endl;

    sprintf(filename, "gtol_%.4u.txt", rank);
    file.open(filename);
    for(ii = 0; ii < kk; ii++) {
        file << theta0[ii] << "\t" << theta1[ii] << "\t" << kTheta[ii] << endl;
    }
    file.close();

    VecDestroy(&ui);
    VecDestroy(&ke);
    VecDestroy(&kel);

    delete topo;
    delete geom;
    delete sw;

    delete[] els;
    delete[] xi0;
    delete[] xi1;
    delete[] theta0;
    delete[] theta1;
    delete[] kTheta;

    PetscFinalize();

    return 0;
}
