#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1] 

f = open(filename,"r")
size = 0
for line in f:
	size = size + 1

print size

A = np.zeros((size,size),dtype=np.float64)

f = open(filename,"r")
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
			#A[row][col] = val
			if val > +1.0e-10:
				A[row][col] = +1.0
			elif val < -1.0e-10:
				A[row][col] = -1.0

	row = row + 1

levs=np.linspace(0.0,0.1,100)
#plt.contourf(A,levs,interpolation='none')
plt.imshow(A,interpolation='nearest',vmin=-1.0,vmax=+1.0,cmap='coolwarm')
#plt.colorbar()
plt.savefig('mat.png')
plt.show()
