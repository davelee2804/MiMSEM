#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

nt = 10

lambda_r = np.zeros((nt,8))
lambda_i = np.zeros((nt,8))
timescale = np.zeros((nt,8))
num_non_zeros = np.zeros(nt)

for ii in np.arange(nt):
	index = 0
	for jj in np.arange(8):
		filename = "output/dmd_mode_" + str(100+4*ii) + "_0" + str(jj) + ".meta"
		if os.path.isfile(filename):
			A = np.loadtxt(filename)
			if A[1] > 1.0e-8 and A[3] > 0.0 and A[3] < 12.0:
				lambda_r[ii][index] = A[0]
				lambda_i[ii][index] = A[1]
				timescale[ii][index] = A[3]
				index = index + 1
				num_non_zeros[ii] = index

lambda_r_s = np.zeros((nt,8))
lambda_i_s = np.zeros((nt,8))
timescale_s = np.zeros((nt,8))
for ii in np.arange(nt):
	nnz = num_non_zeros[ii]
	inds = np.argsort(lambda_r[ii][:nnz])
	lambda_r_s[ii][:nnz] = lambda_r[ii][inds]
	lambda_i_s[ii][:nnz] = lambda_i[ii][inds]
	timescale_s[ii][:nnz] = timescale[ii][inds]

plt.plot(np.arange(nt)+100,timescale_s[:,0],'-o')
plt.plot(np.arange(nt)+100,timescale_s[:,1],'-o')
plt.show()
