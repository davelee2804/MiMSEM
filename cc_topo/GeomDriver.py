#!/usr/bin/env python

import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

from Proc2 import *
from Geom2 import *

pn = int(sys.argv[1])
ne = int(sys.argv[2])
n_procs = int(sys.argv[3])
xg, yg, zg = init_geom(pn, ne, True)

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(xg, yg, zg, c='k')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()

pc = ParaCube(n_procs,pn,ne)
for pi in np.arange(n_procs):
	proc = pc.procs[pi]
	coords = np.zeros((proc.n0g,3),dtype=np.float64)
	for ii in np.arange(proc.n0g):
		coords[ii,0] = xg[proc.loc0[ii]]
		coords[ii,1] = yg[proc.loc0[ii]]
		coords[ii,2] = zg[proc.loc0[ii]]

	np.savetxt('geom_%.4u'%pi + '.txt', coords, fmt='%.18e')
