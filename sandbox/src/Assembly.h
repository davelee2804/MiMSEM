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

class Uhmat {
    public:
        Uhmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Uhmat();
        double* UtQUflat;
        double** UtQU;
        double** UtQV;
        double** VtQU;
        double** VtQV;
        double** Qaa;
        double** Qab;
        double** Qbb;
        double** Ut;
        double** Vt;
        double** UtQaa;
        double** UtQab;
        double** VtQba;
        double** VtQbb;
        Wii* Q;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
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
        Vec vlInv;
        Vec vgInv;
        void assemble();
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

class WtQUmat {
    public:
        WtQUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~WtQUmat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        double* WtQUflat;
        double* WtQVflat;
        double** Wt;
        double** Qaa;
        double** Qab;
        double** WtQaa;
        double** WtQab;
        double** WtQU;
        double** WtQV;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        M2_j_xy_i* W;
        Wii* Q;
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
        double* UtQUflat;
        double** Ut;
        double** Vt;
        double** Qab;
        double** Qba;
        double** UtQab;
        double** VtQba;
        double** UtQV;
        double** VtQU;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
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

class Whmat {
    public:
        Whmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~Whmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec h2);
};
