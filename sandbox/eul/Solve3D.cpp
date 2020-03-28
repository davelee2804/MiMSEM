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
    quad = new GaussLobatto(_topo->elOrd);
    node = new LagrangeNode(_topo->elOrd, quad);
    edge = new LagrangeEdge(_topo->elOrd, node);
    M1 = new Umat(_topo, _geom, node, edge);

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
    ISCreateGeneral(MPI_COMM_WORLD, geom->nk*topo->n1, inds_g, PETSC_COPY_VALUES, &is_g);

    VecScatterCreate(v_g, is_g, v_l, is_l, &scat);
    delete[] inds_g;

    VecDestroy(&v_l);
    VecDestroy(&v_g);
    ISDestroy(&is_l);
    ISDestroy(&is_g);

    MatCreate(MPI_COMM_WORLD, &M);
    MatSetSizes(M, lSize, lSize, gSize, gSize);
    MatSetType(M, MATMPIAIJ);
    MatMPIAIJSetPreallocation(M, 2*3*2*2*(elOrd)*(elOrd+1), PETSC_NULL, 2*3*2*2*(elOrd)*(elOrd+1), PETSC_NULL);
    MatZeroEntries(M);

    if(!rank) cout << "Solve3D() - dt: " << dt << ", del2: " << del2 << endl;

    // assemble the matrix
    M1->assemble(0, SCALE, false);
    MatGetOwnershipRange(M1->M, &mi, &mf);
    for(int kk = 0; kk < geom->nk; kk++) {
        if(kk > 0 && kk < geom->nk-1) {
            //dzInv = -0.25*(geom->thick[kk-1][0] + geom->thick[kk][0]);

            //dzInv = (1.0/geom->thick[kk-1][0]/geom->thick[kk-1][0])/(0.5*(geom->thick[kk-1][0] + geom->thick[kk+0][0]));
            dzInv = (-16.0/geom->thick[kk][0]/geom->thick[kk-1][0])/(geom->thick[kk-1][0] + geom->thick[kk+0][0]);
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

            //dzInv  = (1.0/geom->thick[kk][0]/geom->thick[kk][0])/(0.5*(geom->thick[kk-1][0] + geom->thick[kk+0][0]));
            dzInv = (-16.0/geom->thick[kk][0]/geom->thick[kk+0][0])/(geom->thick[kk-1][0] + geom->thick[kk+0][0]);
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

        dzInv  = (2.0/geom->thick[kk][0]);
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
            //dzInv = -0.25*(geom->thick[kk][0] + geom->thick[kk+1][0]);

            //dzInv  = (1.0/geom->thick[kk][0]/geom->thick[kk][0])/(0.5*(geom->thick[kk+1][0] + geom->thick[kk+0][0]));
            dzInv = (-16.0/geom->thick[kk][0]/geom->thick[kk+0][0])/(geom->thick[kk+1][0] + geom->thick[kk+0][0]);
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

            //dzInv = (1.0/geom->thick[kk+1][0]/geom->thick[kk+1][0])/(0.5*(geom->thick[kk+1][0] + geom->thick[kk+0][0]));
            dzInv = (-16.0/geom->thick[kk][0]/geom->thick[kk+1][0])/(geom->thick[kk+1][0] + geom->thick[kk+0][0]);
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

    VecSet(x, 1.0);
    MatMult(M, x, b);
    VecDot(x, b, &dzInv);
    if(!rank) cout << "Solve3D() - |M|: " << dzInv << endl;

    KSPCreate(MPI_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M, M);

    KSPGetPC(ksp, &pc);
    KSPSetType(ksp, KSPGMRES);
    PCSetType(pc, PCBJACOBI);
    PCBJacobiSetTotalBlocks(pc, geom->nk*size*topo->nElsX*topo->nElsX, NULL);
    KSPSetUp(ksp);
    PCBJacobiGetSubKSP(pc, &nlocal, &first_local, &subksp);

    for(int ii = 0; ii < nlocal; ii++) {
        KSPGetPC(subksp[ii], &subpc);
        PCSetType(subpc, PCLU);
    }
    KSPSetTolerances(ksp, 1.0e-16, 1.0e-50, PETSC_DEFAULT, 1000);
    KSPSetOptionsPrefix(ksp, "ksp_imp_visc_vert_");
    KSPSetFromOptions(ksp);

    ug = new Vec[geom->nk];
    for(int kk = 0; kk < geom->nk; kk++) {
        VecCreateMPI(MPI_COMM_WORLD, topo->n1l, topo->nDofs1G, &ug[kk]);
    }
}

void Solve3D::Solve(Vec* bg, Vec* xg) {
    double norm;

    for(int kk = 0; kk < geom->nk; kk++) {
        MatMult(M1->M, bg[kk], ug[kk]);
        VecScale(ug[kk], 2.0/geom->thick[kk][0]);
    }

    RepackVector(ug, b);
    VecNorm(b, NORM_2, &norm);
    if(!rank) cout << "Solve3D() - |b|: " << norm << endl;
    KSPSolve(ksp, b, x);
    VecNorm(x, NORM_2, &norm);
    if(!rank) cout << "Solve3D() - |x|: " << norm << endl;
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
    for(int kk = 0; kk < geom->nk; kk++) {
        VecDestroy(&ug[kk]);
    }
    delete[] ug;

    delete quad;
    delete node;
    delete edge;
    delete M1;
}

void Solve3D::RepackVector(Vec* ux, Vec _v) {
    int shift;
    PetscScalar *uArray, *vArray;

    VecGetArray(vl, &vArray);
    for(int kk = 0; kk < geom->nk; kk++) {
        VecScatterBegin(topo->gtol_1, ux[kk], ul, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(  topo->gtol_1, ux[kk], ul, INSERT_VALUES, SCATTER_FORWARD);

        shift = kk * topo->n1;
        VecGetArray(ul, &uArray);
        for(int ii = 0; ii < topo->n1; ii++) {
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
        shift = kk * topo->n1;
        VecGetArray(ul, &uArray);
        for(int ii = 0; ii < topo->n1; ii++) {
            uArray[ii] = vArray[shift+ii];
        }
        VecRestoreArray(ul, &uArray);

        VecZeroEntries(ux[kk]);
        VecScatterBegin(topo->gtol_1, ul, ux[kk], INSERT_VALUES, SCATTER_REVERSE); // insert NOT add
        VecScatterEnd(  topo->gtol_1, ul, ux[kk], INSERT_VALUES, SCATTER_REVERSE);
    }
    VecRestoreArray(vl, &vArray);
}
