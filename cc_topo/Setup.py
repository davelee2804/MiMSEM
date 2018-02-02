#!/usr/bin/env python

import os
import sys
import numpy as np
from Proc2 import *
from Geom2 import *

# Input arguments: <polynomial_order> <num_elements_per_dim_per_face> <num_processors>
# NOTE: num_processors MUST be 6.n^2 for positive integer n
pn = int(sys.argv[1])
ne = int(sys.argv[2])
n_procs = int(sys.argv[3])

os.makedirs('./src2/input')

# Generate the topology
pc = ParaCube(n_procs,pn,ne)

for pi in np.arange(n_procs):
	pc.print_nodes(pi)
	pc.print_edges(pi,0)
	pc.print_edges(pi,1)
	pc.print_faces(pi)
	pc.procs[pi].writeLocalSizes()

# Write the grid metadata
f = open('grid_res.txt', 'w')
f.write(str(pn) + '\n')
f.write(str(ne/pc.npx))
f.close()

# Generate the geometry
xg, yg, zg = init_geom(pn, ne, False, True)

for pi in np.arange(n_procs):
	proc = pc.procs[pi]
	coords = np.zeros((proc.n0g,3),dtype=np.float64)
	for ii in np.arange(proc.n0g):
		coords[ii,0] = xg[proc.loc0[ii]]
		coords[ii,1] = yg[proc.loc0[ii]]
		coords[ii,2] = zg[proc.loc0[ii]]

	np.savetxt('./src2/input/geom_%.4u'%pi + '.txt', coords, fmt='%.18e')

