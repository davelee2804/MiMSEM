#include <math.h>
#include <stdlib.h>
#include <iostream>

#include <mpi.h>

#include "Basis.h"

using namespace std;

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

    if(n == 1) {
        x[0] = -1.0; x[1] = +1.0;
        w[0] = 1.0; w[1] = 1.0;
    }
    else if(n == 2) {
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
        a = 2.0*sqrt(7.0)/21.0;
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
    else if(n == 7) {
        x[0] = -1.0; x[1] = -0.871740148509607; x[2] = -0.591700181433142; x[3] = -0.209299217902479;
        x[4] = +0.209299217902479; x[5] = +0.591700181433142; x[6] = +0.871740148509607; x[7] = +1.0;
        w[0] = w[7] = 0.035714285714286; w[1] = w[6] = 0.210704227143506; w[2] = w[5] = 0.341122692483504; w[3] = w[4] = 0.412458794658704;
    }
    else {
        cout << "invalid gauss-lobatto quadrature order: " << n << endl;
    }

    a = 0.0;
    for(ii = 0; ii <= n; ii++) {
        a+= w[ii];
    }
    if(fabs(a - 2.0) > 1.0e-8) {
        cout << "quadrature weights error!" << endl;
    }
}

GaussLobatto::~GaussLobatto() {
    delete[] x;
    delete[] w;
}

LagrangeNode::LagrangeNode(int _n, GaussLobatto* _q) {
    int ii, jj;
    GaussLobatto* q_tmp = new GaussLobatto(_n);

    n = _n;
    q = _q;

    a = new double[n+1];
    x = new double[n+1];
    for(ii = 0; ii <= n; ii++) {
        x[ii] = q_tmp->x[ii];
    }

    // evaluate the coefficients
    for(ii = 0; ii <= n; ii++) {
        a[ii] = 1.0;
        for(jj = 0; jj <= n; jj++) {
            if(jj == ii) continue;
            a[ii] *= 1.0/(q->x[ii] - q->x[jj]);
        }
    }

    // evaluate the lagrange basis function at quadrature points matrix
    ljxi = new double*[q->n+1];
    for(ii = 0; ii <= q->n; ii++) {
        ljxi[ii] = new double[n+1];
        for(jj = 0; jj <= n; jj++) {
            //ljxi[ii][jj] = eval(q->x[ii], jj);
            ljxi[ii][jj] = eval_q(q->x[ii], jj);
        }
    }

    // ...and the transpose
    ljxi_t = new double*[n+1];
    for(ii = 0; ii <= n; ii++) {
        ljxi_t[ii] = new double[q->n+1];
        for(jj = 0; jj <= q->n; jj++) {
            //ljxi_t[ii][jj] = eval(q->x[jj], ii);
            ljxi_t[ii][jj] = eval_q(q->x[jj], ii);
        }
    }

    delete q_tmp;
}

LagrangeNode::~LagrangeNode() {
    int ii;

    delete[] a;
    delete[] x;

    for(ii = 0; ii <= q->n; ii++) {
        delete[] ljxi[ii];
    }
    delete[] ljxi;

    for(ii = 0; ii <= n; ii++) {
        delete[] ljxi_t[ii];
    }
    delete[] ljxi_t;
}

double LagrangeNode::eval(double _x, int i) {
    int jj;
    double p = 1.0;

    for(jj = 0; jj <= n; jj++) {
        if(jj == i) continue;
        p *= _x - q->x[jj];
    }

    return a[i]*p;
}

// evaluate at arbitrary location (not necessarily gll node)
double LagrangeNode::eval_q(double _x, int i) {
    double y = 1.0;
    for(int jj = 0; jj <= n; jj++) {
        if(jj == i) continue;
        y *= (_x - x[jj])/(x[i] - x[jj]);
    }
    return y;
}

double LagrangeNode::evalDeriv(double _x, int ii) {
    int jj, kk;
    double aa, bb;

    bb = 0.0;

    for(jj = 0; jj <= n; jj++) {
        if(jj == ii) continue;

        aa = 1.0;
        for(kk = 0; kk <= n; kk++) {
            if(kk == ii) continue;
            if(kk == jj) continue;

            aa *= (_x - x[kk])/(x[ii] - x[kk]);
        }

        bb += aa/(x[ii] - x[jj]);
    }

    return bb;
}

void LagrangeNode::test() {
    int ii, jj, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(!rank) {
        for(ii = 0; ii <= n; ii++) {
            for(jj = 0; jj <= n; jj++) {
                cout << eval(x[ii],jj) << "\t";
            }
            cout << endl;
        }
        cout << endl;

        for(ii = 0; ii <= n; ii++) {
            for(jj = 0; jj <= n; jj++) {
                cout << evalDeriv(x[ii],jj) << "\t";
            }
            cout << endl;
        }
    }
}

LagrangeEdge::LagrangeEdge(int _n, LagrangeNode* _l) {
    int ii, jj;

    n = _n;
    l = _l;

    // evaluate the edge basis function at quadrature points matrix
    ejxi = new double*[l->q->n+1];
    for(ii = 0; ii <= l->q->n; ii++) {
        ejxi[ii] = new double[n];
        for(jj = 0; jj < n; jj++) {
            ejxi[ii][jj] = eval(l->q->x[ii], jj);
        }
    }

    // ...and the transpose
    ejxi_t = new double*[n];
    for(ii = 0; ii < n; ii++) {
        ejxi_t[ii] = new double[l->q->n+1];
        for(jj = 0; jj <= l->q->n; jj++) {
            ejxi_t[ii][jj] = eval(l->q->x[jj], ii);
        }
    }
}

LagrangeEdge::~LagrangeEdge() {
    int ii;

    for(ii = 0; ii <= l->q->n; ii++) {
        delete[] ejxi[ii];
    }
    delete[] ejxi;

    for(ii = 0; ii < n; ii++) {
        delete[] ejxi_t[ii];
    }
    delete[] ejxi_t;
}

double LagrangeEdge::eval(double x, int i) {
    int jj;
    double c = 0.0;

    for(jj = 0; jj <= i; jj++) {
        c -= l->evalDeriv(x, jj);
    }

    return c;
}
