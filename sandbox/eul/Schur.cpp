#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "Schur.h"

using namespace std;

Schur::Schur(Topo* _topo, Geom* _geom) {
    int elOrd2 = _topo->elOrd * _topo->elOrd;
    int lSize = _geom->nk * _topo->n2l;
    int gSize = _geom->nk * _topo->nDofs2G;
    int index;
    int *inds, *inds_g;
    PC pc;
    Vec v_l, v_g;
    IS is_l, is_g;

    topo = _topo;
    geom = _geom;

    VecCreateSeq(MPI_COMM_SELF, lSize, &vl);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &x);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &b);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, lSize, lSize, gSize, gSize);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
    MatZeroEntries(M);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M, M);
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetType(ksp, KSPGMRES);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, elOrd2, NULL);
    KSPSetOptionsPrefix(ksp, "ksp_schur_");
    KSPSetFromOptions(ksp);

    // create the scatter
    index = 0;
    inds_g = new int[lSize];
    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            inds = topo->elInds2_g(ei%topo->nElsX, ei/topo->nElsX);
            for(int ii = 0; ii < elOrd2; ii++) {
                inds_g[index++] = kk*topo->nDofs2G + inds[ii];
            }
        }
    }
    
    VecCreateSeq(MPI_COMM_SELF, lSize, &v_l);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &v_g);
    ISCreateStride(MPI_COMM_SELF, lSize, 0, 1, &is_l);
    ISCreateGeneral(MPI_COMM_WORLD, lSize, inds_g, PETSC_COPY_VALUES, &is_g);

    VecScatterCreate(v_g, is_g, v_l, is_l, &scat);

    delete[] inds_g;
    VecDestroy(&v_l);
    VecDestroy(&v_g);
    ISDestroy(&is_l);
    ISDestroy(&is_g);
}

void Schur::AddFromVertMat(int ei, Mat Az) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int nCols, row_g, cols_g[999];
    int* inds = topo->elInds2_g(ei%topo->nElsX, ei/topo->nElsX);
    const int* cols;
    const double *vals;

    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < elOrd2; ii++) {
            row_g = kk*topo->nDofs2G + inds[ii];

            MatGetRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
                cols_g[cc] = (cols[cc]/elOrd2)*topo->nDofs2G + inds[cols[cc]%elOrd2];
            }
            MatSetValues(M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
            MatRestoreRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
        }
    }
}

void Schur::AddFromHorizMat(int kk, Mat Ax) {
    int mi, mf, nCols, row_g, cols_g[999];
    const int* cols;
    const double *vals;

    MatGetOwnershipRange(Ax, &mi, &mf);

    for(int mm = mi; mm < mf; mm++) {
        MatGetRow(Ax, mm, &nCols, &cols, &vals);
        row_g = kk*topo->nDofs2G + mm;
        for(int cc = 0; cc < nCols; cc++) {
            cols_g[cc] = kk*topo->nDofs2G + cols[cc];
        }
        MatSetValues(M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
        MatRestoreRow(Ax, mm, &nCols, &cols, &vals);
    }
}

void Schur::RepackFromVert(Vec* vz, Vec v) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int ind_g;
    int* inds;
    PetscScalar *vzArray, *vArray;

    VecZeroEntries(vl);
    VecGetArray(vl, &vArray);
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
        VecGetArray(vz[ei], &vzArray);
        for(int kk = 0; kk < geom->nk; kk++) {
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = kk*topo->n2l + inds[ii];
                vArray[ind_g] = vzArray[kk*elOrd2+ii];
            }
        }
        VecRestoreArray(vz[ei], &vzArray);
    }
    VecRestoreArray(vl, &vArray);

    // scatter to the global vector
    VecScatterBegin(scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
}

// note: vx are assumed to be local vectors
void Schur::RepackFromHoriz(Vec* vx, Vec v) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int ind_g;
    int* inds;
    PetscScalar *vxArray, *vArray;

    VecZeroEntries(vl);
    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecGetArray(vx[kk], &vxArray);
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = kk*topo->n2l + inds[ii];
                vArray[ind_g] = vxArray[inds[ii]];
            }
        }
        VecRestoreArray(vx[kk], &vxArray);
    }
    VecRestoreArray(vl, &vArray);

    // scatter to the global vector
    VecScatterBegin(scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
    VecScatterEnd(  scat, vl, v, ADD_VALUES, SCATTER_REVERSE);
}

// note: vx are assumed to be local vectors
void Schur::UnpackToHoriz(Vec v, Vec* vx) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int ind_g;
    int* inds;
    PetscScalar *vxArray, *vArray;

    VecScatterBegin(scat, v, vl, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(  scat, v, vl, INSERT_VALUES, SCATTER_FORWARD);

    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecGetArray(vx[kk], &vxArray);
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
            for(int ii = 0; ii < elOrd2; ii++) {
                ind_g = kk*topo->n2l + inds[ii];
                vxArray[inds[ii]] = vArray[ind_g];
            }
        }
        VecRestoreArray(vx[kk], &vxArray);
    }
    VecRestoreArray(vl, &vArray);
}

Schur::~Schur() {
    VecDestroy(&vl);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&M);
    KSPDestroy(&ksp);
    VecScatterDestroy(&scat);
}
