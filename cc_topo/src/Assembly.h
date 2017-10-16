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
