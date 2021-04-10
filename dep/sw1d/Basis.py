import numpy as np
from scipy.special import legendre

def P7(x):
	a = -35.0*x + 315.0*np.power(x,3) - 693.0*np.power(x,5) + 429.0*np.power(x,7)
	return a/16.0

def P8(x):
	a = 35.0 - 1260.0*np.power(x,2) + 6930.0*np.power(x,4) - 12012.0*np.power(x,6) + 6435.0*np.power(x,8)
	return a/128.0

# exact for polynomial of degree 2n - 1
class GaussLobatto:
	def __init__(self,n):
		self.n = n
		if n == 1:
			self.x = np.array([-1.0,+1.0])
			self.w = np.array([1.0,1.0])
		elif n == 2:
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
			self.x = np.array([-1.0,-0.8717401485096066153375,-0.5917001814331423021445,-0.2092992179024788687687,\
                                 0.2092992179024788687687,0.5917001814331423021445,0.8717401485096066153375,1.0])
			self.w = np.array([0.03571428571428571428571,0.210704227143506039383,0.3411226924835043647642,0.4124587946587038815671,\
                               0.412458794658703881567,0.341122692483504364764,0.210704227143506039383,0.03571428571428571428571])
		elif n == 8:
			self.x = np.array([-1.0,-0.8997579954114601573124,-0.6771862795107377534459,-0.3631174638261781587108,0.0,\
                               0.3631174638261781587108,0.6771862795107377534459,0.8997579954114601573124,1.0])
			self.w = np.array([0.02777777777777777777778,0.1654953615608055250463,0.274538712500161735281,0.3464285109730463451151,0.3715192743764172335601,\
                               0.3464285109730463451151,0.2745387125001617352807,0.165495361560805525046,0.02777777777777777777778])
		else:
			print('invalid gauss-lobatto quadrature order: ' + str(n))

		if np.abs(np.sum(self.w) - 2.0) > 1.0e-8:
			print('quadrature weights error!')

class LagrangeNode:
	def __init__(self,n,m):
		self.n = n # polynomial order
		self.m = m # quadrature order
		self.quad = GaussLobatto(m)
		self.node = GaussLobatto(n)

		self.M_ij_c = np.zeros((self.m+1,self.n+1),dtype=np.float64)
		for i in np.arange(self.m+1):
			for j in np.arange(self.n+1):
				self.M_ij_c[i,j] = self.eval(self.quad.x[i],self.node.x,j)

	def eval(self,x,xi,j):
		y = 1.0
		for i in np.arange(len(xi)):
			if i == j:
				continue
			y = y*(x-xi[i])/(xi[j]-xi[i])

		return y

	def eval_deriv(self,x,xi,j):
		d = 0.0
		for i in np.arange(len(xi)):
			if i == j:
				continue
			a = 1.0/(xi[j]-xi[i])
			for m in np.arange(len(xi)):
				if m == i or m == j:
					continue
				a = a*(x-xi[m])/(xi[j]-xi[m])

			d = d + a

		return d

class LagrangeEdge:
	def __init__(self,n,m):
		self.n = n # polynomial order
		self.m = m # quadrature order
		self.node = LagrangeNode(n,m)

		self.M_ij_c = np.zeros((m+1,n),dtype=np.float64)
		for i in np.arange(m+1):
			for j in np.arange(n):
				for k in np.arange(j+1):
					#self.M_ij_c[i,j] = self.M_ij_c[i,j] - self.node.eval_deriv(self.node.quad.x[i],self.node.quad.x,k)
					self.M_ij_c[i,j] = self.M_ij_c[i,j] - self.node.eval_deriv(self.node.quad.x[i],self.node.node.x,k)

	def eval(self,x,i):
		c = 0.0
		for j in np.arange(i+1):
			#c = c - self.node.eval_deriv(x,self.node.quad.x,j)
			c = c - self.node.eval_deriv(x,self.node.node.x,j)

		return c
