class Umat {
    public:
        Umat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble(double scale);
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
        void assemble(Vec h2, double scale);
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

//////////////
class Krhs {
    public:
        Krhs(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Krhs();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        void assemble(Vec k, Vec* F);
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

class W0mat {
    public:
        W0mat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~W0mat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class U0mat {
    public:
        U0mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~U0mat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class WU0mat {
    public:
        WU0mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~WU0mat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble();
};

class W0hmat {
    public:
        W0hmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~W0hmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec h);
};

class UtQh_vec {
    public:
        UtQh_vec(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~UtQh_vec();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Vec ul;
        Vec hl;
        Vec ug;
        void assemble(Vec h2);
};

class WtUmat {
    public:
        WtUmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~WtUmat();
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
        void assemble();
};

class U_up_mat {
    public:
        U_up_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~U_up_mat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec ui);
};

class W_up_mat {
    public:
        W_up_mat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~W_up_mat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec ui);
};
class Uh_up_mat {
    public:
        Uh_up_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Uh_up_mat();
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
        void assemble(Vec h2, Vec u1);
};

class Ph_vec {
    public:
        Ph_vec(Topo* _topo, Geom* _geom, LagrangeNode* _l);
        ~Ph_vec();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        Wii* Q;
        PetscScalar* entries;
        Vec vl;
        Vec vg;
        Vec vlInv;
        Vec vgInv;
        void assemble(Vec h2);
};
