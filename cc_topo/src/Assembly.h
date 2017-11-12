class Umat {
    public:
        Umat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class Wmat {
    public:
        Wmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~Wmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class Pmat {
    public:
        Pmat(Topo* _topo, Geom* _geom, LagrangeNode* _l);
        ~Pmat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        Mat M;
        void assemble();
};

class Uhmat {
    public:
        Uhmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Uhmat();
        double* ck;
        double* UtQUflat;
        double** JxU;
        double** JxV;
        double** JyU;
        double** JyV;
        double** JxUt;
        double** JyVt;
        double** UtQ;
        double** VtQ;
        double** UtQU;
        double** UtQV;
        double** VtQU;
        double** VtQV;
        Wii* Q;
        JacM1* J;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        M1x_j_Fxy_i* Uh;
        M1y_j_Fxy_i* Vh;
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec h2);
};

class Pvec {
    public:
        Pvec(Topo* _topo, Geom* _geom, LagrangeNode* _l);
        ~Pvec();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        Wii* Q;
        PetscScalar* entries;
        Vec vl;
        Vec vg;
        void assemble();
};

class Phvec {
    public:
        Phvec(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Phvec();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        double* ck;
        PetscScalar* entries;
        Vec vl;
        Vec vg;
        void assemble(Vec h2);
};

class WtQmat {
    public:
        WtQmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~WtQmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class PtQmat {
    public:
        PtQmat(Topo* _topo, Geom* _geom, LagrangeNode* _l);
        ~PtQmat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        Mat M;
        void assemble();
};

class UtQmat {
    public:
        UtQmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~UtQmat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class PtQUmat {
    public:
        PtQUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~PtQUmat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        double* ckx;
        double* cky;
        double* PtQUflat;
        double* PtQVflat;
        double** PtQ;
        M1x_j_Exy_i* U;
        M1y_j_Exy_i* V;
        M0_j_xy_i* P;
        void assemble(Vec q1);
};

class WtQUmat {
    public:
        WtQUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~WtQUmat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        double* ckx;
        double* cky;
        double* WtQUflat;
        double* WtQVflat;
        double** JxU;
        double** JxV;
        double** JyU;
        double** JyV;
        double** JW;
        double** JWt;
        double** WtQ;
        double** WtQU;
        double** WtQV;
        M1x_j_Cxy_i* U;
        M1y_j_Cxy_i* V;
        M2_j_xy_i* W;
        Wii* Q;
        JacM1* J1;
        JacM2* J2;
        void assemble(Vec u1);
};

class RotMat {
    public:
        RotMat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~RotMat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        double* ckx;
        double* cky;
        double* UtQUflat;
        double** JxU;
        double** JxV;
        double** JyU;
        double** JyV;
        double** JxUt;
        double** JyVt;
        double** UtQ;
        double** VtQ;
        double** UtQU;
        double** UtQV;
        double** VtQU;
        double** VtQV;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        M1x_j_Dxy_i* Uq;
        M1y_j_Dxy_i* Vq;
        JacM1* J;
        Wii* Q;
        void assemble(Vec q0);
};

class E10mat {
    public:
        E10mat(Topo* _topo);
        ~E10mat();
        Topo* topo;
        Mat E10;
        Mat E01;
};

class E21mat {
    public:
        E21mat(Topo* _topo);
        ~E21mat();
        Topo* topo;
        Mat E21;
        Mat E12;
};
