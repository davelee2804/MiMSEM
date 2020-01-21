#include <iostream>

#include <petsc.h>
#include <petscis.h>
#include <petscvec.h>
#include <petscmat.h>

#include "LinAlg.h"
#include "Basis.h"
#include "Topo.h"
#include "Geom.h"
#include "L2Vecs.h"
#include "Schur.h"

using namespace std;

#define VERTICALLY_CONTIGUOUS 1

Schur::Schur(Topo* _topo, Geom* _geom) {
    int elOrd2 = _topo->elOrd * _topo->elOrd;
    int lSize = _geom->nk * _topo->n2l;
    int gSize = _geom->nk * _topo->nDofs2G;
    int index;
    int *inds, *inds_g;
    Vec v_l, v_g;
    IS is_l, is_g;

    topo = _topo;
    geom = _geom;

    elOrd = _topo->elOrd;
    nElsX = _topo->nElsX;
    inds2 = new int[elOrd2];

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    VecCreateSeq(MPI_COMM_SELF, lSize, &vl);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &x);
    VecCreateMPI(MPI_COMM_WORLD, lSize, gSize, &b);

    // create the scatter
    index = 0;
    inds_g = new int[lSize];
#ifdef VERTICALLY_CONTIGUOUS
    for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
        for(int kk = 0; kk < geom->nk; kk++) {
            for(int ii = 0; ii < elOrd2; ii++) {
                inds_g[index++] = rank*(geom->nk*topo->n2l) + ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
            }
        }
    }
#else
    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ei = 0; ei < topo->nElsX*topo->nElsX; ei++) {
            inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
            for(int ii = 0; ii < elOrd2; ii++) {
                inds_g[index++] = rank * (geom->nk*topo->n2l) + kk*topo->n2l + inds[ii];
            }
        }
    }
#endif
    
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

void Schur::InitialiseMatrix() {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int lSize = geom->nk * topo->n2l;
    int gSize = geom->nk * topo->nDofs2G;

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, lSize, lSize, gSize, gSize);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 11*elOrd2*elOrd2, PETSC_NULL, 11*elOrd2*elOrd2, PETSC_NULL);
    MatZeroEntries(M);

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M, M);
}

void Schur::DestroyMatrix() {
    MatDestroy(&M);
    KSPDestroy(&ksp);
}

void Schur::AddFromVertMat(int ei, Mat Az) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int nCols, row_g, cols_g[999];
#ifdef VERTICALLY_CONTIGUOUS
    int lShift = rank*(geom->nk*topo->n2l) + ei*geom->nk*elOrd2;
#else
    int* inds = topo->elInds2_l(ei%topo->nElsX, ei/topo->nElsX);
#endif
    const int* cols;
    const double *vals;

    for(int kk = 0; kk < geom->nk; kk++) {
        for(int ii = 0; ii < elOrd2; ii++) {
#ifdef VERTICALLY_CONTIGUOUS
            row_g = lShift + kk*elOrd2 + ii;
#else
            row_g = rank*(geom->nk*topo->n2l) + kk*topo->n2l + inds[ii];
#endif

            MatGetRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
            for(int cc = 0; cc < nCols; cc++) {
#ifdef VERTICALLY_CONTIGUOUS
                cols_g[cc] = lShift + cols[cc];
#else
                cols_g[cc] = rank*(geom->nk*topo->n2l) + (cols[cc]/elOrd2)*topo->n2l + inds[cols[cc]%elOrd2];
#endif
            }
            MatSetValues(M, 1, &row_g, nCols, cols_g, vals, ADD_VALUES);
            MatRestoreRow(Az, kk*elOrd2+ii, &nCols, &cols, &vals);
        }
    }
}

void Schur::AddFromHorizMat(int kk, Mat Ax) {
    int elOrd2 = topo->elOrd * topo->elOrd;
    int mi, mf, nCols, row_g, cols_g[999], rank_i;
    const int* cols;
    const double *vals;

    MatGetOwnershipRange(Ax, &mi, &mf);

    for(int mm = mi; mm < mf; mm++) {
        MatGetRow(Ax, mm, &nCols, &cols, &vals);

        rank_i = mm/topo->n2l;
#ifdef VERTICALLY_CONTIGUOUS
        row_g = rank_i*geom->nk*topo->n2l + ((mm-mi)/elOrd2)*geom->nk*elOrd2 + kk*elOrd2 + (mm-mi)%elOrd2;
#else
        row_g = rank_i*geom->nk*topo->n2l + kk*topo->n2l + mm%topo->n2l;
#endif
        if(mm<rank*topo->n2l || mm>=(rank+1)*topo->n2l) cout << "SCHUR: matrix assebly error [1]! " << rank << ", " << mm << endl;
	if(rank != rank_i                             ) cout << "SCHUR: matrix assebly error [2]! " << rank << ", " << mm << endl;

        for(int cc = 0; cc < nCols; cc++) {
            rank_i = cols[cc]/topo->n2l;
#ifdef VERTICALLY_CONTIGUOUS
            cols_g[cc] = /*rank_i*geom->nk*topo->n2l +*/ (cols[cc]/elOrd2)*geom->nk*elOrd2 + kk*elOrd2 + cols[cc]%elOrd2;
#else
            cols_g[cc] = rank_i*geom->nk*topo->n2l + kk*topo->n2l + cols[cc]%topo->n2l;
#endif
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
#ifdef VERTICALLY_CONTIGUOUS
                ind_g = ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
#else
                ind_g = kk*topo->n2l + inds[ii];
#endif
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
#ifdef VERTICALLY_CONTIGUOUS
                ind_g = ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
#else
                ind_g = kk*topo->n2l + inds[ii];
#endif
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
#ifdef VERTICALLY_CONTIGUOUS
                ind_g = ei*geom->nk*elOrd2 + kk*elOrd2 + ii;
#else
                ind_g = kk*topo->n2l + inds[ii];
#endif
                vxArray[inds[ii]] = vArray[ind_g];
            }
        }
        VecRestoreArray(vx[kk], &vxArray);
    }
    VecRestoreArray(vl, &vArray);
}

void Schur::Solve(L2Vecs* d_exner) {
    int nlocal, first_local, size;
    PC pc, subpc;
    KSP* subksp;

    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  M, MAT_FINAL_ASSEMBLY);

#ifdef VERTICALLY_CONTIGUOUS
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
#else
    KSPGetPC(ksp, &pc);
    KSPSetType(ksp, KSPGMRES);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, geom->nk, NULL);
    KSPSetUp(ksp);
    PCBJacobiGetSubKSP(pc, &nlocal, &first_local, &subksp);

    for(int ii = 0; ii < nlocal; ii++) {
        KSPGetPC(subksp[ii], &subpc);
        PCSetType(subpc, PCJACOBI);
        KSPSetType(subksp[ii], KSPGMRES);
        KSPSetTolerances(subksp[ii], 1.e-16, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    }
#endif
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetOptionsPrefix(ksp, "ksp_schur_");
    KSPSetFromOptions(ksp);

    KSPSolve(ksp, b, x);
    UnpackToHoriz(x, d_exner->vl);
}

Schur::~Schur() {
    delete[] inds2;
    VecDestroy(&vl);
    VecDestroy(&x);
    VecDestroy(&b);
    VecScatterDestroy(&scat);
}
