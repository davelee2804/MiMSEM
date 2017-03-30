import numpy as np
from scipy.special import legendre

def P7(x):
	a = -35.0*x + 315.0*np.power(x,3) - 693.0*np.power(x,5) + 429.0*np.power(x,7)
	return a/16.0

# exact for polynomial of degree 2n - 1
class GaussLobatto:
	def __init__(self,n):
		self.n = n
		if n == 2:
			self.x = np.array([-1.0,0.0,+1.0])
			self.w = np.array([1.0,4.0,1.0])/3.0
		elif n == 3:
			self.x = np.array([-1.0,-np.sqrt(0.2),+np.sqrt(0.2),+1.0])
			self.w = np.array([1.0,5.0,5.0,1.0])/6.0
		elif n == 4:
			self.x = np.array([-1.0,-np.sqrt(3.0/7.0),0.0,+np.sqrt(3.0/7.0),+1.0])
			self.w = np.array([9.0,49.0,64.0,49.0,9.0])/90.0
		elif n == 5:
			a = 2.0*np.sqrt(7.0)/21.0
			self.x = np.array([-1.0,-np.sqrt(1.0/3.0+a),-np.sqrt(1.0/3.0-a),+np.sqrt(1.0/3.0-a),+np.sqrt(1.0/3.0+a),+1.0])
			self.w = np.array([2.0,14.0-np.sqrt(7.0),14.0+np.sqrt(7.0),14.0+np.sqrt(7.0),14.0-np.sqrt(7.0),2.0])/30.0
		elif n == 6:
			a = 2.0*np.sqrt(5.0/3.0)/11.0
			self.x = np.array([-1.0,-np.sqrt(5.0/11.0+a),-np.sqrt(5.0/11.0-a),0.0,+np.sqrt(5.0/11.0-a),+np.sqrt(5.0/11.0+a),+1.0])
			self.w = np.array([1.0/21.0,(124.0-7.0*np.sqrt(15.0))/350.0,(124.0+7.0*np.sqrt(15.0))/350.0,256.0/525.0,(124.0+7.0*np.sqrt(15.0))/350.0,(124.0-7.0*np.sqrt(15.0))/350.0,1.0/21.0])
		elif n == 7:
			phi = np.arccos(np.sqrt(55.0)/30.0)
			r3 = np.power(320.0*np.sqrt(55.0)/265837.0,1.0/3.0)
			x1 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 5.0/13.0))
			x2 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 5.0/13.0 + 4.0*np.pi/3.0))
			x3 = np.sqrt(2.0*r3*np.cos(phi/3.0 + 5.0/13.0 + 2.0*np.pi/3.0))
			w1 = 1.0/28.0/(P7(x1)*P7(x1))
			w2 = 1.0/28.0/(P7(x1)*P7(x2))
			w3 = 1.0/28.0/(P7(x1)*P7(x3))
			self.x = np.array([-1.0,-x1,-x2,-x3,+x3,+x2,+x1,+1.0])
			self.w = np.array([1.0/28.0,w1,w2,w3,w3,w2,w1,1.0/28.0])
		else:
			print 'invalid gauss-lobatto quadrature order: ' + str(n)

		if np.abs(np.sum(self.w) - 2.0) > 1.0e-8:
			print 'quadrature weights error!'

# exact for polynomial of degree 2n + 1
class GaussLegendre:
	def __init__(self,n):
		self.n = n
		if n == 2:
			self.x = np.array([-np.sqrt(1.0/3.0),+np.sqrt(1.0/3.0)])
			self.w = np.array([1.0,1.0])
		elif n == 3:
			self.x = np.array([-np.sqrt(0.6),0.0,+np.sqrt(0.6)])
			self.w = np.array([5.0,8.0,5.0])/9.0
		elif n == 4:
			a = (2.0/7.0)*np.sqrt(1.2)
			self.x = np.array([-np.sqrt(3.0/7.0+a),-np.sqrt(3.0/7.0-a),+np.sqrt(3.0/7.0-a),+np.sqrt(3.0/7.0+a)])
			self.w = np.array([18.0-np.sqrt(30.0),18.0+np.sqrt(30.0),18.0+np.sqrt(30.0),18.0-np.sqrt(30.0)])/36.0
		elif n == 5:
			a = 2.0*np.sqrt(10.0/7.0)
			self.x = np.array([-np.sqrt(5.0+a)/3.0,-np.sqrt(5.0-a)/3.0,0.0,+np.sqrt(5.0-a)/3.0,+np.sqrt(5.0+a)/3.0])
			self.w = np.array([322.0-13.0*np.sqrt(70.0),322.0+13.0*np.sqrt(70),512.0,322.0+13.0*np.sqrt(70.0),322.0-13.0*np.sqrt(70)])/900.0
		else:
			print 'invalid gauss-legendre quadrature order: ' + str(n)

		if np.abs(np.sum(self.w) - 2.0) > 1.0e-8:
			print 'quadrature weights error!'

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
