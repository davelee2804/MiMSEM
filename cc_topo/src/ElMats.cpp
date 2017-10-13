#include "Basis.h"
#include "ElMats.h"

void mult(int ni, int nj, int nk, double** A, double** B, double** C) {
    int ii, jj, kk;

    for(ii = 0; ii < ni; ii++) {
        for(jj = 0; jj < nj; jj++) {
            C[ii,jj] = 0.0;
            for(kk = 0; kk < nk; kk++) {
                C[ii,jj] += A[ii,kk]*B[kk,jj];
            }
        }
    }
}

// Outer product of 0-form in x and 1-form in y (columns)
// evaluated at the Gauss Lobatto quadrature points (rows)
// n: basis function order
// m: quadrature point order
M1x_j_xy_i::M1x_j_xy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii, jj;
    int mi, nj, nn, np1, mp1;
    double li, ei;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = l->ljxi[ii%mp1,jj%np1];
            ei = n->ejli[ii/mp1,jj/np1];
            A[ii,jj] = li*ei;
        }
    }
}

M1x_j_xy_i::~M1x_j_xy_i() {
    int mp1 = l->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// Outer product of 1-form in x and 0-form in y (columns)
// evaluated at the Gauss Lobatto quadrature points (rows)
// n: basis function order
// m: quadrature point order
M1y_j_xy_i::M1y_j_xy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii, jj;
    int mi, nj, nn, np1, mp1;
    double li, ei;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }

    for(jj = 0; jj < nj; jj++) {
        for(ii = 0; ii < mi; ii++) {
            li = l->ljxi[ii/mp1,jj/nn];
            ei = e->ejxi[ii%mp1,jj%nn];
            A[ii,jj] = ei*li;
        }
    }
}

M1y_j_xy_i::~M1y_j_xy_i() {
    int mp1 = l->q->n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// As above but with 1 forms interpolated to quadrature points
// (normal) velocity interpolated to quadrature points
M1x_j_Cxy_i::M1x_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

void M1x_j_Cxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei, ck;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ck += c[kk]*l.ljxi[ii%mp1,kk%np1]*e.ejxi[ii/mp1,kk/np1];
        }

        for(jj = 0; jj < nj; jj++) {
            li = l.ljxi[ii%mp1,jj%np1];
            ei = e.ejxi[ii/mp1,jj/np1];
            A[ii,jj] = ck*li*ei;
        }
    }
}

M1x_j_Cxy_i::~M1x_j_Cxy_i() {
    int mp1 = l.q.n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// As above but with 1 forms interpolated to quadrature points
// (normal) velocity interpolated to quadrature points
M1y_j_Cxy_i::M1y_j_Cxy_i(LagrangeNode* _l, LagrangeEdge* _e) {
    int ii;
    int mi, nj, nn, np1, mp1;

    l = _l;
    e = _e;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    A = new double*[mi];
    for(ii = 0; ii < mi; ii++) {
        A[ii] = new double[nj];
    }
}

void M1y_j_Cxy_i::assemble(double* c) {
    int ii, jj, kk;
    int mi, nj, nn, np1, mp1;
    double li, ei;

    nn = l->n;
    np1 = nn + 1;
    mp1 = l->q->n + 1;
    mi = mp1*mp1;
    nj = np1*nn;

    for(ii = 0; ii < mi; ii++) {
        ck = 0.0;
        for(kk = 0; kk < nj; kk++) {
            ck += c[kk]*e.ejxi[ii%mp1,kk%nn]*l.ljxi[ii/mp1,kk/nn];
        }

        for(jj = 0; jj < nj; jj++) {
            ei = e.ejxi[ii%mp1,jj%nn];
            li = l.ljxi[ii/mp1,jj/nn];
            A[ii,jj] = ck*ei*li;
        }
    }
}

M1y_j_Cxy_i::~M1y_j_Cxy_i() {
    int mp1 = l.q.n + 1;
    int ii;

    for(ii = 0; ii < mp1*mp1; ii++) {
        delete[] A[ii];
    }
    delete[] A;
}

// As above but with 1 forms cross product interpolated to quadrature points
// (tangent) velocity interpolated to quadrature points
class M1x_j_Exy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		n = self.n
		np1 = self.n+1
		mp1 = self.m+1
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(nj):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.N[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Mjy = self.E[i/mp1,j/np1]

				self.A[i,j] = Njx*Mjy*ck

		return self.A

# As above but with 0 forms interpolated to quadrature points
# potential vorticity interpolated to quadrature points
class M1x_j_Dxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	#def assemble(self,c,d):
	def assemble(self,c):
		np1 = self.n + 1
		mp1 = self.m + 1
		n = self.n
		n2 = np1*np1
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.N[i%mp1,k%np1]*self.N[i/mp1,k/np1]

			#dk = 0.0
			#for k in np.arange(n*n):
			#	dk = dk + d[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Mjy = self.E[i/mp1,j/np1]

				self.A[i,j] = Njx*Mjy*ck#/dk

		return self.A

# thickness interpolated to quadrature points
# for diagnosis of hu
class M1x_j_Fxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		np1 = self.n + 1
		mp1 = self.m + 1
		n = self.n
		n2 = n*n
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Mjy = self.E[i/mp1,j/np1]

				self.A[i,j] = Njx*Mjy*ck

		return self.A


# As above but with 1 forms (cross product) interpolated to quadrature points
# (tangent) velocity interpolated to quadrature points
class M1y_j_Exy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		np1 = self.n+1
		mp1 = self.m+1
		n = self.n
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(nj):
				ck = ck + c[k]*self.N[i%mp1,k%np1]*self.E[i/mp1,k/np1]

			for j in np.arange(nj):
				Mjx = self.E[i%mp1,j%n]
				Njy = self.N[i/mp1,j/n]

				self.A[i,j] = Mjx*Njy*ck

		return self.A

# As above but with 0 forms interpolated to quadrature points
# potential vorticity interpolated to quadrature points
class M1y_j_Dxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		n2 = n*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)


	#def assemble(self,c,d):
	def assemble(self,c):
		n = self.n
		np1 = self.n + 1
		mp1 = self.m + 1
		n2 = np1*np1
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.N[i%mp1,k%np1]*self.N[i/mp1,k/np1]

			#dk = 0.0
			#for k in np.arange(n*n):
			#	dk = dk + d[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Mjx = self.E[i%mp1,j%n]
				Njy = self.N[i/mp1,j/n]

				self.A[i,j] = Mjx*Njy*ck#/dk

		return self.A

# thickness interpolated to quadrature points
# for diagnosis of hv
class M1y_j_Fxy_i:
	def __init__(self,n,m):
		self.n = n
		self.m = m

		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*n
		n2 = n*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((mp1,np1),dtype=np.float64)
		for j in np.arange(np1):
			for i in np.arange(mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(mp1):
				self.E[i,j] = edge.eval(q.x[i],j)


	def assemble(self,c):
		n = self.n
		np1 = self.n + 1
		mp1 = self.m + 1
		n2 = n*n
		mi = mp1*mp1
		nj = np1*self.n

		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Mjx = self.E[i%mp1,j%n]
				Njy = self.N[i/mp1,j/n]

				self.A[i,j] = Mjx*Njy*ck

		return self.A

# Outer product of 1-form in x and 1-form in y
# evaluated at the Gauss Lobatto quadrature points
# n: basis function order
# m: quadrature point order
class M2_j_xy_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = n*n
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Mj = LagrangeEdge(n)
		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i%mp1]
				y = q.x[i/mp1]
				Mjx = Mj.eval(x,j%n)
				Mjy = Mj.eval(y,j/n)
				self.A[i,j] = Mjx*Mjy

# 0 form basis function terms (j) evaluated at the
# quadrature points (i)
class M0_j_xy_i:
	def __init__(self,n,m):
		np1 = n+1
		mp1 = m+1
		mi = mp1*mp1
		nj = np1*np1
		q = GaussLobatto(m)
		self.A = np.zeros((mi,nj),dtype=np.float64)
		Nj = LagrangeNode(n)
		for j in np.arange(nj):
			for i in np.arange(mi):
				x = q.x[i%mp1]
				y = q.x[i/mp1]
				Njx = Nj.eval(x,j%np1)
				Njy = Nj.eval(y,j/np1)
				self.A[i,j] = Njx*Njy

class M0_j_Cxy_i:
	def __init__(self,n,m):
		self.n = n
		self.n2 = n*n
		self.np1 = n+1
		self.mp1 = m+1
		self.mi = self.mp1*self.mp1
		self.nj = self.np1*self.np1
		q = GaussLobatto(m)
		self.A = np.zeros((self.mi,self.nj),dtype=np.float64)

		node = LagrangeNode(n)
		self.N = np.zeros((self.mp1,self.np1),dtype=np.float64)
		for j in np.arange(self.np1):
			for i in np.arange(self.mp1):
				self.N[i,j] = node.eval(q.x[i],j)

		edge = LagrangeEdge(n)
		self.E = np.zeros((self.mp1,n),dtype=np.float64)
		for j in np.arange(n):
			for i in np.arange(self.mp1):
				self.E[i,j] = edge.eval(q.x[i],j)

	def assemble(self,c):
		n = self.n
		n2 = self.n2
		mi = self.mi
		nj = self.nj
		mp1 = self.mp1
		np1 = self.np1
		
		for i in np.arange(mi):
			ck = 0.0
			for k in np.arange(n2):
				ck = ck + c[k]*self.E[i%mp1,k%n]*self.E[i/mp1,k/n]

			for j in np.arange(nj):
				Njx = self.N[i%mp1,j%np1]
				Njy = self.N[i/mp1,j/np1]
				self.A[i,j] = Njx*Njy*ck

		return self.A

# Quadrature weights diagonal matrix
class Wii:
	def __init__(self,m):
		mp1 = m+1
		mi = mp1*mp1
		mj = mp1*mp1
		q = GaussLobatto(m)
		self.A = np.zeros((mi,mj),dtype=np.float64)
		for i in np.arange(mi):
			self.A[i,i] = q.w[i%mp1]*q.w[i/mp1]
