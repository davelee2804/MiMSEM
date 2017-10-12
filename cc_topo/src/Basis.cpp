#include <math.h>
#include <iostream>

#include "Basis.h"

double P7(double x) {
    double a = -35.0*x + 315.0*pow(x,3) - 693.0*pow(x,5) + 429.0*pow(x,7);
    return a/16.0;
}

double P8(double x) {
    double a = 35.0 - 1260.0*pow(x,2) + 6930.0*pow(x,4) - 12012.0*pow(x,6) + 6435.0*pow(x,8);
    return a/128.0;
}

// exact for polynomial of degree 2n - 1
GaussLobatto::GaussLobatto(int _n) {
    int ii;
    double a;

    n = _n;

    x = new double[n+1];
    w = new double[n+1];

    if(n == 2) {
        x[0] = -1.0; x[1] = 0.0; x[2] = +1.0;
        w[0] = 1.0/3.0; w[1] = 4.0/3.0; w[2] = 1.0/3.0;
    }
    else if(n == 3) {
        x[0] = -1.0; x[1] = -sqrt(0.2); x[2] = +sqrt(0.2); x[3] = +1.0;
        w[0] = w[3] = 1.0/6.0; w[1] = w[2] = 5.0/6.0;
    }
    else if(n == 4) {
        x[0] = -1.0; x[1] = -sqrt(3.0/7.0); x[2] = 0.0; x[3] = +sqrt(3.0/7.0); x[4] = +1.0;
        w[0] = w[4] = 0.1; w[1] = w[3] = 49.0/90.0; w[2] = 64.0/90.0;
    }
    else if(n == 5) {
        a = 2.0*np.sqrt(7.0)/21.0;
        x[0] = -1.0; x[1] = -sqrt(1.0/3.0+a); x[2] = -sqrt(1.0/3.0-a); x[3] = +sqrt(1.0/3.0-a); x[4] = +sqrt(1.0/3.0+a); x[5] = +1.0;
        w[0] = w[5] = 1.0/15.0; w[1] = w[4] = (14.0-sqrt(7.0))/30.0; w[2] = w[3] = (14.0+sqrt(7.0))/30.0;
    }
    else if(n == 6) {
        a = 2.0*sqrt(5.0/3.0)/11.0;
        x[0] = -1.0; x[1] = -sqrt(5.0/11.0+a); x[2] = -sqrt(5.0/11.0-a); x[3] = 0.0; x[4] = +sqrt(5.0/11.0-a); x[5] = +sqrt(5.0/11.0+a); x[6] = +1.0;
        w[0] = w[6] = 1.0/21.0; w[1] = w[5] = (124.0-7.0*sqrt(15.0))/350.0; w[2] = w[4] = (124.0+7.0*sqrt(15.0))/350.0; w[3] = 256.0/525.0;

////////	elif n == 7:
////////		phi = np.arccos(np.sqrt(55.0)/30.0)
////////		r3 = np.power(320.0*np.sqrt(55.0)/265837.0,1.0/3.0)
////////		x1 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 5.0/13.0))
////////		x2 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 5.0/13.0 + 4.0*np.pi/3.0))
////////		x3 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 5.0/13.0 + 2.0*np.pi/3.0))
////////		w1 = 1.0/28.0/(P7(x1)*P7(x1))
////////		w2 = 1.0/28.0/(P7(x1)*P7(x2))
////////		w3 = 1.0/28.0/(P7(x1)*P7(x3))
////////		self.x = np.array([-1.0,-x1,-x2,-x3,+x3,+x2,+x1,+1.0])
////////		self.w = np.array([1.0/28.0,w1,w2,w3,w3,w2,w1,1.0/28.0])
////////	elif n == 8:
////////		#phi = np.arccos(np.sqrt(91.0)/154)
////////		#r3 = np.power(448.0*np.sqrt(91.0)/570375.0,1.0/3.0)
////////		#x1 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 7.0/15.0))
////////		#x2 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 7.0/15.0 + 4.0*np.pi/3.0))
////////		#x3 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 7.0/15.0 + 2.0*np.pi/3.0))
////////		x1 = 0.899757995411460
////////		x2 = 0.677186279510737
////////		x3 = 0.363117463826178
////////		w1 = 1.0/36.0/(P8(x1)*P8(x1))
////////		w2 = 1.0/36.0/(P8(x2)*P8(x2))
////////		w3 = 1.0/36.0/(P8(x3)*P8(x3))
////////		self.x = np.array([-1.0,-x1,-x2,-x3,0.0,+x3,+x2,+x1,+1.0])
////////		self.w = np.array([1.0/36.0,w1,w2,w3,4096.0/11025.0,w3,w2,w1,1.0/36.0])
    }
    else {
        cout << "invalid gauss-lobatto quadrature order: " << n << endl;
    }

    a = 0.0;
    for(ii = 0; ii <= n; ii++) {
        a+= w[ii];
    }
    if(abs(a - 2.0) > 1.0e-8) {
        cout << "quadrature weights error!" << endl;
    }
}

GaussLobatto::~GaussLobatto() {
    delete[] x;
    delete[] w;
}

class LagrangeNode:
	def __init__(self,n):
		self.n = n
		self.q = GaussLobatto(n)
		self.coeffs()

	def coeffs(self):
		self.a = np.ones(self.n+1)
		for i in np.arange(self.n+1):
			for j in np.arange(self.n+1):
				if j == i:
					continue
				self.a[i] *= 1.0/(self.q.x[i] - self.q.x[j])

	def polyMult(self,a1,a2):
		# number of terms in the input polynomials
		n1 = a1.shape[0]
		n2 = a2.shape[0]
		# order of the input polynomials
		o1 = n1 - 1
		o2 = n2 - 1
		# order of the output polynomial
		o3 = o1 + o2
		# number of terms of the output polynomial
		n3 = o3 + 1
		a3 = np.zeros(n3)
		for i in np.arange(n1):
			for j in np.arange(n2):
				k = i + j
				a3[k] += a1[i]*a2[j]

		return a3

	def polyMultI(self,i,X):
		p2 = np.zeros((self.n+1,2))
		p2[:,0] = 1.0
		for j in np.arange(self.n+1):
			p2[j,1] = -X[j]
		
		pi = np.array([1.0])
		for j in np.arange(self.n+1):
			if j == i:
				continue
			pj = p2[j,:]
			pi = self.polyMult(pi,pj)

		return pi[::-1]

	def eval(self,x,i):
		p = 1.0
		for j in np.arange(self.n+1):
			if j == i:
				continue
			p *= x - self.q.x[j]

		return self.a[i]*p

	def evalDeriv(self,x,i):
		p = self.polyMultI(i,self.q.x)
		dy = 0.0
		for j in np.arange(self.n) + 1:
			dy += j*p[j]*np.power(x,j-1)

		return dy*self.a[i]

class LagrangeEdge:
	def __init__(self,n):
		self.n = n
		self.nf = LagrangeNode(n)

	def eval(self,x,i):
		c = 0.0
		for j in np.arange(i+1):
			c = c - self.nf.evalDeriv(x,j)

		return c
