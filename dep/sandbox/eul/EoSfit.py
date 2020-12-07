#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def evalQuad(x,p3):
	n = len(x)
	y = np.zeros(n)
	for ii in np.arange(n):
		y[ii] = p3[0]*x[ii]*x[ii] + p3[1]*x[ii] + p3[2]

	return y

def evalPi(Theta):
	R = 278.0
	cv = 717.5
	cp = 1004.5
	po = 100000.0
	return cp*np.power(R/po,R/cv)*np.power(Theta,R/cv)
	

R = 278.0
cv = 717.5
cp = 1004.5
po = 100000.0

tMin = 8.
tMax = 350.0
nt = 101

Theta = np.linspace(tMin,tMax,nt)
Pi = cp*np.power(R/po,R/cv)*np.power(Theta,R/cv)

n = 2
pn = np.polyfit(Theta, Pi, n)
print pn
poly = np.poly1d(np.polyfit(Theta, Pi, n))
print poly

plt.plot(Theta,Pi)
plt.plot(Theta,poly(Theta))
plt.show()

plt.plot(Theta,(Pi-poly(Theta))/Pi)
plt.show()

tMin = 2.0
tMax = 400.0

N1 = 33
Theta_1 = np.linspace(tMin,tMax,N1)
N0 = N1-1
Theta_0 = np.zeros(N0)
P3 = np.zeros((N0,3))
x = np.zeros(3)
A = np.zeros((3,3))
for ii in np.arange(N0):
	x[0] = Theta_1[ii];
	x[2] = Theta_1[ii+1];
	x[1] = 0.5*(Theta_1[ii] + Theta_1[ii+1])
	y = cp*np.power(R/po,R/cv)*np.power(x,R/cv)
	A[0][0] = x[0]*x[0]
	A[0][1] = x[0]
	A[0][2] = 1.0
	A[1][0] = x[1]*x[1]
	A[1][1] = x[1]
	A[1][2] = 1.0
	A[2][0] = x[2]*x[2]
	A[2][1] = x[2]
	A[2][2] = 1.0
	Ainv = np.linalg.inv(A);
	P3[ii][:] = np.dot(Ainv,y)

for ii in np.arange(N0):
	Theta_i = np.linspace(Theta_1[ii],Theta_1[ii+1],101,endpoint=True)
	Pi_i = evalQuad(Theta_i, P3[ii][:])
	plt.plot(Theta_i,Pi_i,c='r')

nt = 41
Theta = np.linspace(tMin,tMax,nt)
Pi = cp*np.power(R/po,R/cv)*np.power(Theta,R/cv)

plt.plot(Theta,Pi,c='b')
plt.show()

for ii in np.arange(N0):
	Theta_i = np.linspace(Theta_1[ii],Theta_1[ii+1],101,endpoint=True)
	Pi_n = evalQuad(Theta_i, P3[ii][:])
	Pi_a = evalPi(Theta_i)
	plt.semilogy(Theta_i,np.abs(Pi_a-Pi_n)/Pi_a,c='g')
	print Theta_1[ii], Theta_1[ii+1], P3[ii][:]

plt.show()
