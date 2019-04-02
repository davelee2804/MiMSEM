#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

nk = 30
ztop = 30000.0
mu = 15.0
X = np.zeros(nk+1)
for ii in np.arange(nk+1):
	frac = (1.0*ii)/nk
	X[ii] = ztop*(np.sqrt(mu*frac*frac + 1.0) - 1.0)/(np.sqrt(mu + 1.0) - 1.0)

RD = 287.0
TE = 310.0
TP = 240.0
GRAVITY = 9.80616
OMEGA = 7.29212e-5
T0 = 0.5*(TE+TP)
b = 2.0
KP = 3.0
GAMMA = 0.005
P0 = 100000
RE = 6371229.0
CP = 1004.5

torr_1 = np.zeros(nk+1)
torr_2 = np.zeros(nk+1)
int_torr_1 = np.zeros(nk+1)
int_torr_2 = np.zeros(nk+1)
temp = np.zeros(nk+1)
pres = np.zeros(nk+1)
velx = np.zeros(nk+1)
theta = np.zeros(nk+1)
exner = np.zeros(nk+1)
for ii in np.arange(nk+1):
	A     = 1.0/GAMMA
	B     = (TE - TP)/((TE + TP)*TP)
	C     = 0.5*(KP + 2.0)*(TE - TP)/(TE*TP)
	H     = RD*T0/GRAVITY
	fac   = X[ii]/(b*H)
	fac2  = fac*fac
	cp    = np.cos(2.0*np.pi/9.0)
	cpk   = np.power(cp, KP)
	cpkp2 = np.power(cp, KP+2)
	fac3  = cpk - (KP/(KP+2.0))*cpkp2

	torr_1[ii] = (A*GAMMA/T0)*np.exp(GAMMA*(X[ii])/T0) + B*(1.0 - 2.0*fac2)*np.exp(-fac2)
	torr_2[ii] = C*(1.0 - 2.0*fac2)*np.exp(-fac2)

	int_torr_1[ii] = A*(np.exp(GAMMA*X[ii]/T0) - 1.0) + B*X[ii]*np.exp(-fac2)
	int_torr_2[ii] = C*X[ii]*np.exp(-fac2)

	tempInv = torr_1[ii] - torr_2[ii]*fac3
	temp[ii] = 1.0/tempInv

	pres[ii] = P0*np.exp(-GRAVITY*int_torr_1[ii]/RD + GRAVITY*int_torr_2[ii]*fac3/RD)

	u_mean = (GRAVITY*KP/RE)*int_torr_2[ii]*temp[ii]*(np.power(cp, KP-1.0) - np.power(cp, KP+1.0))
	velx[ii] = -OMEGA*RE*cp + np.sqrt(OMEGA*OMEGA*RE*RE*cp*cp + RE*cp*u_mean)

	exner[ii] = CP*np.power(pres[ii]/P0, RD/CP)
	theta[ii] = temp[ii]*np.power(P0/pres[ii], RD/CP)

thetadExner = np.zeros(nk-1)
for ii in np.arange(nk-1) + 1:
	dExner = (exner[ii+1]-exner[ii-1])/(X[ii+1]-X[ii-1])
	thetadExner[ii-1] = theta[ii]*dExner

#plt.plot(int_torr_1,X,'-o')
#plt.plot(int_torr_2,X,'-o')
plt.plot(temp,X,'-o')
plt.plot(pres/P0,X,'-o')
plt.plot(velx,X,'-o')
plt.plot(exner,X,'-o')
plt.plot(theta,X,'-o')
plt.plot(thetadExner,X[1:-1],'-o')
plt.legend([r'$T$',r'$p/p_0$',r'$u$',r'$\Pi$',r'$\theta$',r'$\theta\Pi_{,z}$'],loc='lower right')
plt.show()
