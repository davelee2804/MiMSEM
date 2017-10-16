class Umat {
    public:
        Umat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat();
        Topo* topo;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class Wmat {
    public:
        Wmat(Topo* _topo, LagrangeEdge* _e);
        ~Wmat();
        Topo* topo;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class Pmat {
    public:
        Pmat(Topo* _topo, LagrangeNode* _l);
        ~Pmat();
        Topo* topo;
        LagrangeNode* l;
        Mat M;
        void assemble();
};

class Uhmat {
    public:
        Uhmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~Uhmat();
        double* ck;
        double* UtQUflat;
        double* VtQVflat;
        double** UtQ;
        double** VtQ;
        M1x_j_Fxy_i* Uh;
        M1y_j_Fxy_i* Vh;
        Topo* topo;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec h2);
};

class Pvec {
    public:
        Pvec(Topo* _topo, LagrangeNode* _l);
        ~Pvec();
        Topo* topo;
        LagrangeNode* l;
        PetscScalar* entries;
        Vec v;
        void assemble();
};

class Phvec {
    public:
        Phvec(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~Phvec();
        Topo* topo;
        LagrangeNode* l;
        LagrangeEdge* e;
        double* ck;
        PetscScalar* entries;
        Vec v;
        void assemble(Vec h2);
};

class WtQmat {
    public:
        WtQmat(Topo* _topo, LagrangeEdge* _e);
        ~WtQmat();
        Topo* topo;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class PtQmat {
    public:
        PtQmat(Topo* _topo, LagrangeNode* _l);
        ~PtQmat();
        Topo* topo;
        LagrangeNode* l;
        Mat M;
        void assemble();
};

class UtQmat {
    public:
        UtQmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~UtQmat();
        Topo* topo;
        LagrangeNode* l;
        LagrangeNode* e;
        Mat M;
        void assemble();
};

class PtQUmat {
    public:
        PtQUmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~PtQUmat();
        Topo* topo;
        LagrangeNode* l;
        LagrangeNode* e;
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
        WtQUmat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~WtQUmat();
        Topo* topo;
        LagrangeNode* l;
        LagrangeNode* e;
        Mat M;
        double* ckx;
        double* cky;
        double* WtQUflat;
        double* WtQVflat;
        double** WtQ;
        M1x_j_Cxy_i* U;
        M1y_j_Cxy_i* V;
        M2_j_xy_i* W;
        void assemble(Vec u1);
};

class RotMat {
    public:
        RotMat(Topo* _topo, LagrangeNode* _l, LagrangeEdge* _e);
        ~RotMat();
        Topo* topo;
        LagrangeNode* l;
        LagrangeNode* e;
        Mat M;
        double* ckx;
        double* cky;
        double* UtQVflat;
        double* VtQUflat;
        double** UtQ;
        double** VtQ;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        M1x_j_Dxy_i* Uq;
        M1y_j_Dxy_i* Vq;
        void assemble(Vec q0);
};
