#!/usr/bin/env python3

import sys
import numpy as np

num_proc = int(sys.argv[1])
path = sys.argv[2]

# get the array sizes
filename = path + '/edges_x_0000.txt'
a = np.loadtxt(filename)
n = a.shape[0]
filename = path + '/grid_res.txt'
a = np.loadtxt(filename)
el_ord = int(a[0])
n_els_x = int(a[1])
print("el ord:  %u"%el_ord)
print("n els x: %u"%n_els_x)

inds_glob = np.zeros((num_proc,2,n),dtype=np.int32)

n_dofs_x = el_ord*n_els_x
n_dofs_1 = 2*num_proc*n_dofs_x*n_dofs_x

inds_skel = np.zeros(n_dofs_1,dtype=np.int32)
inds_dual = np.zeros((num_proc,2,n),dtype=np.int32)
inds_intl = np.zeros((num_proc,2,n),dtype=np.int32)
inds_skel[:] = -1
inds_dual[:,:,:] = -1
inds_intl[:,:,:] = -1

# populate the global indices
print("loading edge dofs...")
for ii in np.arange(num_proc):
    filename = path + '/edges_x_%.4u.txt'%ii
    a = np.loadtxt(filename)
    b = np.array(a,dtype=np.int32)
    inds_glob[ii,0,:] = b

    filename = path + '/edges_y_%.4u.txt'%ii
    a = np.loadtxt(filename)
    b = np.array(a,dtype=np.int32)
    inds_glob[ii,1,:] = b

# check for common indices accross processors, and
# add these to the set of global skeleton indices
print("generating skeleton dofs...")
glob_ind = 0
for ii in np.arange(2*num_proc-1):
    print("proc i: %u"%ii)
    vi = ii%2
    pi = ii//2
    for ni in np.arange(n):
        gi = inds_glob[pi,vi,ni]
        for jj in np.arange(ii+1,2*num_proc):
            vj = jj%2
            pj = jj//2
            for nj in np.arange(n):
                gj = inds_glob[pj,vj,nj]
                if gi == gj and inds_skel[gi] == -1:
                    inds_skel[gi] = glob_ind
                    glob_ind += 1

filename = path + '/global_to_skeleton.txt'
np.savetxt(filename,inds_skel,fmt='%u')

# generate the associated dual indices, dofs within
# the same element as a skeleton index, in the same 
# direction
print("generating dual dofs...\n")
n_dofs_el = el_ord*(el_ord+1)
inds_l_x = np.zeros(n_dofs_el,dtype=np.int32)
inds_l_y = np.zeros(n_dofs_el,dtype=np.int32)
inds_g_x = np.zeros(n_dofs_el,dtype=np.int32)
inds_g_y = np.zeros(n_dofs_el,dtype=np.int32)
for proc_i in np.arange(num_proc):
    locl_ind = 0
    for ex in np.arange(n_els_x):
        for ey in np.arange(n_els_x):
            # global indices in the x direction in element (x,y)
            kk = 0
            is_skel = False
            for iy in np.arange(el_ord):
                for ix in np.arange(el_ord+1):
                    inds_l_x[kk] = (ey*el_ord + iy)*(n_dofs_x + 1) + ex*el_ord + ix
                    inds_g_x[kk] = inds_glob[proc_i,0,(ey*el_ord + iy)*(n_dofs_x + 1) + ex*el_ord + ix]
                    if inds_skel[inds_g_x[kk]] > -1:
                        is_skel = True
                    kk += 1
            if is_skel:
                for kk in np.arange(n_dofs_el):
                    if inds_skel[inds_g_x[kk]] == -1:
                        inds_dual[proc_i,0,inds_l_x[kk]] = locl_ind
                        locl_ind += 1

            # global indices in the y direction in element (x,y)
            kk = 0
            is_skel = False
            for iy in np.arange(el_ord+1):
                for ix in np.arange(el_ord):
                    inds_l_y[kk] = (ey*el_ord + iy)*(n_dofs_x) + ex*el_ord + ix
                    inds_g_y[kk] = inds_glob[proc_i,1,(ey*el_ord + iy)*(n_dofs_x) + ex*el_ord + ix]
                    if inds_skel[inds_g_y[kk]] > -1:
                        is_skel = True
                    kk += 1
            if is_skel:
                for kk in np.arange(n_dofs_el):
                    if inds_skel[inds_g_y[kk]] == -1:
                        inds_dual[proc_i,1,inds_l_y[kk]] = locl_ind
                        locl_ind += 1

    filename = path + '/dual_inds_x_%.4u.txt'%proc_i
    np.savetxt(filename,inds_dual[proc_i,0,:],fmt='%u')
    filename = path + '/dual_inds_y_%.4u.txt'%proc_i
    np.savetxt(filename,inds_dual[proc_i,1,:],fmt='%u')

# generate the internal (local) degrees of freedom (ie: everything else)
print("generating internal dofs...\n")
for proc_i in np.arange(num_proc):
    locl_ind = 0
    for ni in np.arange(n):
        ind_x = inds_glob[proc_i,0,ni]
        if inds_skel[ind_x] == -1 and inds_dual[proc_i,0,ni] == -1:
            inds_intl[proc_i,0,ni] = locl_ind
            locl_ind += 1
        ind_y = inds_glob[proc_i,1,ni]
        if inds_skel[ind_y] == -1 and inds_dual[proc_i,1,ni] == -1:
            inds_intl[proc_i,1,ni] = locl_ind
            locl_ind += 1

    filename = path + '/internal_inds_x_%.4u.txt'%proc_i
    np.savetxt(filename,inds_intl[proc_i,0,:],fmt='%u')
    filename = path + '/internal_inds_y_%.4u.txt'%proc_i
    np.savetxt(filename,inds_intl[proc_i,1,:],fmt='%u')
