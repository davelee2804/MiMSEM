#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

f = open("mat.dat","r")
size = 0
for line in f:
	size = size + 1

print size

A = np.zeros((size,size),dtype=np.float64)

f = open("mat.dat","r")
row = 0
for line in f:
	line2 = line.split("(")
	for line3 in line2:
		line4 = line3.split(", ")
		if len(line4) > 1:
			col = int(line4[0])
			valstr = line4[1].split(')')
			val = float(valstr[0])
			#if row == 0:
			#	print col, val
			#A[row][col] = np.abs(val)
			A[row][col] = val

	row = row + 1

levs=np.linspace(0.0,0.1,100)
#plt.contourf(A,levs,interpolation='none')
plt.imshow(A,interpolation='none',vmin=-0.1,vmax=+0.1)
plt.colorbar()
plt.show()
