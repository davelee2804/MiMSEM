#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscksp.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "L2Vecs.h"
#include "ElMats.h"
#include "VertOps.h"
#include "Assembly.h"
#include "Solve3D.h"

#define SCALE 1.0e+8

using namespace std;

Solve3D::Solve3D(Topo* _topo, Geom* _geom, double dt, double del2) {
    int lSize = _geom->nk * _topo->n1l;
    int gSize = _geom->nk * _topo->nDofs1G;
    int index;
    int *inds_g;
    int mi, mf, nCols, row_g, col_g[999];
    const int* cols;
    const double *vals;
    double val_g[999], dzInv;
    int nlocal, first_local, size;
    PC pc, subpc;
    KSP* subksp;
    Vec v_l, v_g;
    IS is_l, is_g;
    GaussLobatto* quad = new GaussLobatto(_topo->elOrd);
    LagrangeNode* node = new LagrangeNode(_topo->elOrd, quad);
    LagrangeEdge* edge = new LagrangeEdge(_topo->elOrd, node);
    Umat* M1 = new Umat(topo, geom, node, edge);

    topo = _topo;
    geom = _geom;

    elOrd = _topo->elOrd;
    nElsX = _topo->nElsX;
    elOrd2 = elOrd*elOrd;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    VecCreateSeq(MPI_COMM_SELF, topo->n1, &ul);
    VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->n1, &vl);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &x);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &b);

    // create the scatter
    index = 0;
    inds_g = new int[geom->nk*topo->n1];
    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < topo->n1; ii++) {
            inds_g[index++] = kk*topo->nDofs1G + topo->loc1[ii];
        }
    }
    
    VecCreateSeq(MPI_COMM_SELF, geom->nk*topo->n1, &v_l);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &v_g);
    ISCreateStride(MPI_COMM_SELF, geom->nk*topo->n1, 0, 1, &is_l);
    ISCreateGeneral(MPI_COMM_WORLD, lSize, inds_g, PETSC_COPY_VALUES, &is_g);

    VecScatterCreate(v_g, is_g, v_l, is_l, &scat);
    delete[] inds_g;

    VecDestroy(&v_l);
    VecDestroy(&v_g);
    ISDestroy(&is_l);
    ISDestroy(&is_g);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, lSize, lSize, gSize, gSize);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 3*2*2*(elOrd)*(elOrd+1), PETSC_NULL, 3*2*2*(elOrd)*(elOrd+1), PETSC_NULL);
    MatZeroEntries(M);

    // assemble the matrix
    M1->assemble(0, SCALE, false);
    MatGetOwnershipRange(M1->M, &mi, &mf);
    for(int kk = 0; kk < geom->nk; kk++) {
        if(kk > 0 && kk < geom->nk-1) {
            dzInv = (1.0/geom->thick[kk-1][0]/geom->thick[kk-1][0])/(0.5*(geom->thick[kk-1][0] + geom->thick[kk+0][0]));
            for(int mm = mi; mm < mf; mm++) {
                MatGetRow(M1->M, mm, &nCols, &cols, &vals);
                row_g = kk * topo->nDofs1G + mm;
                for(int col_i = 0; col_i < nCols; col_i++) {
                    col_g[col_i] = (kk-1) * topo->nDofs1G + cols[col_i];
                    val_g[col_i] = dt * del2 * dzInv * vals[col_i];
                }
                MatSetValues(M, 1, &row_g, nCols, col_g, val_g, ADD_VALUES);
                MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
            }

            dzInv  = (1.0/geom->thick[kk][0]/geom->thick[kk][0])/(0.5*(geom->thick[kk-1][0] + geom->thick[kk+0][0]));
            for(int mm = mi; mm < mf; mm++) {
                MatGetRow(M1->M, mm, &nCols, &cols, &vals);
                row_g = kk * topo->nDofs1G + mm;
                for(int col_i = 0; col_i < nCols; col_i++) {
                    col_g[col_i] = kk * topo->nDofs1G + cols[col_i];
                    val_g[col_i] = -dt * del2 * dzInv * vals[col_i];
                }
                MatSetValues(M, 1, &row_g, nCols, col_g, val_g, ADD_VALUES);
                MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
            }
        }

        dzInv  = (1.0/geom->thick[kk][0]);
        for(int mm = mi; mm < mf; mm++) {
            MatGetRow(M1->M, mm, &nCols, &cols, &vals);
            row_g = kk * topo->nDofs1G + mm;
            for(int col_i = 0; col_i < nCols; col_i++) {
                col_g[col_i] = kk * topo->nDofs1G + cols[col_i];
                val_g[col_i] = dzInv * vals[col_i];
            }
            MatSetValues(M, 1, &row_g, nCols, col_g, val_g, ADD_VALUES);
            MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
        }

        if(kk > 0 && kk < geom->nk-1) {
            dzInv  = (1.0/geom->thick[kk][0]/geom->thick[kk][0])/(0.5*(geom->thick[kk+1][0] + geom->thick[kk+0][0]));
            for(int mm = mi; mm < mf; mm++) {
                MatGetRow(M1->M, mm, &nCols, &cols, &vals);
                row_g = kk * topo->nDofs1G + mm;
                for(int col_i = 0; col_i < nCols; col_i++) {
                    col_g[col_i] = kk * topo->nDofs1G + cols[col_i];
                    val_g[col_i] = -dt * del2 * dzInv * vals[col_i];
                }
                MatSetValues(M, 1, &row_g, nCols, col_g, val_g, ADD_VALUES);
                MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
            }

            dzInv = (1.0/geom->thick[kk+1][0]/geom->thick[kk+1][0])/(0.5*(geom->thick[kk+1][0] + geom->thick[kk+0][0]));
            for(int mm = mi; mm < mf; mm++) {
                MatGetRow(M1->M, mm, &nCols, &cols, &vals);
                row_g = kk * topo->nDofs1G + mm;
                for(int col_i = 0; col_i < nCols; col_i++) {
                    col_g[col_i] = (kk+1) * topo->nDofs1G + cols[col_i];
                    val_g[col_i] = dt * del2 * dzInv * vals[col_i];
                }
                MatSetValues(M, 1, &row_g, nCols, col_g, val_g, ADD_VALUES);
                MatRestoreRow(M1->M, mm, &nCols, &cols, &vals);
            }
        }
    }
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M, M);

    KSPGetPC(ksp, &pc);
    KSPSetType(ksp, KSPGMRES);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, size*topo->nElsX*topo->nElsX, NULL);
    KSPSetUp(ksp);
    PCBJacobiGetSubKSP(pc, &nlocal, &first_local, &subksp);

    for(int ii = 0; ii < nlocal; ii++) {
        KSPGetPC(subksp[ii], &subpc);
        PCSetType(subpc, PCLU);
    }
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetOptionsPrefix(ksp, "ksp_schur_");
    KSPSetFromOptions(ksp);

    delete M1;
    delete quad;
    delete node;
    delete edge;
}

void Solve3D::Solve(Vec* bg, Vec* xg) {
    RepackVector(bg, b);
    KSPSolve(ksp, b, x);
    UnpackVector(x, xg);
}

Solve3D::~Solve3D() {
    VecDestroy(&ul);
    VecDestroy(&vl);
    VecDestroy(&x);
    VecDestroy(&b);
    VecScatterDestroy(&scat);
    MatDestroy(&M);
    KSPDestroy(&ksp);
}

void Solve3D::RepackVector(Vec* ux, Vec _v) {
    int shift;
    PetscScalar *uArray, *vArray;

    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_1, ux[kk], ul, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, ux[kk], ul, INSERT_VALUES, SCATTER_FORWARD);

        shift = kk * topo->n1l;
        VecGetArray(ul, &uArray);
        for(int ii = 0; ii < topo->n1l; ii++) {
            vArray[shift+ii] = uArray[ii];
        }
        VecRestoreArray(ul, &uArray);
    }
    VecRestoreArray(vl, &vArray);

    VecZeroEntries(_v);
    VecScatterBegin(scat, vl, _v, INSERT_VALUES, SCATTER_REVERSE); // insert NOT add
    VecScatterEnd(  scat, vl, _v, INSERT_VALUES, SCATTER_REVERSE);
}

void Solve3D::UnpackVector(Vec _v, Vec* ux) {
    int shift;
    PetscScalar *uArray, *vArray;

    VecScatterBegin(scat, _v, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  scat, _v, vl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        shift = kk * topo->n1l;
        VecGetArray(ul, &uArray);
        for(int ii = 0; ii < topo->n1l; ii++) {
            uArray[ii] = vArray[shift+ii];
        }
        VecRestoreArray(ul, &uArray);

        VecZeroEntries(ux[kk]);
        VecScatterBegin(topo->gtol_1, ul, ux[kk], INSERT_VALUES, SCATTER_REVERSE); // insert NOT add
        VecScatterEnd(  topo->gtol_1, ul, ux[kk], INSERT_VALUES, SCATTER_REVERSE);
    }
    VecRestoreArray(vl, &vArray);
}
