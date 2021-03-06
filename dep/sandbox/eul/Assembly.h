class Umat {
    public:
        Umat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale, bool vert_scale);
};

class Wmat {
    public:
        Wmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~Wmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale, bool vert_scale);
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
        void assemble(Vec h2, int lev, bool const_vert, double scale);
        void assemble_up(Vec h2, int lev, double scale, Vec u1);
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
        void assemble(int lev, double scale);
};

class Phvec {
    public:
        Phvec(Topo* _topo, Geom* _geom, LagrangeNode* _l);
        ~Phvec();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        Wii* Q;
        PetscScalar* entries;
        Vec vl;
        Vec vg;
        void assemble(Vec hl, int lev, double scale);
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
        void assemble(Vec u1, int lev, double scale);
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
        void assemble(Vec q0, int lev, double scale);
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
        void assemble(Vec rho, int lev, double scale, bool vert_scale_rho);
};

class Ut_mat {
    public:
        Ut_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Ut_mat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale);
};

class UtQWmat {
    public:
        UtQWmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~UtQWmat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        double* UtQWflat;
        double* VtQWflat;
        double** Ut;
        double** Vt;
        double** Qaa;
        double** Qba;
        double** UtQaa;
        double** VtQba;
        double** UtQW;
        double** VtQW;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        M2_j_xy_i* W;
        Wii* Q;
        void assemble(Vec u1, double scale);
};

class WtQdUdz_mat {
    public:
        WtQdUdz_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~WtQdUdz_mat();
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
        void assemble(Vec u1, double scale);
};

class EoSvec {
    public:
        EoSvec(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~EoSvec();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        double** Wt;
        double** WtQ;
        M2_j_xy_i* W;
        Wii* Q;
        Vec vl;
        Vec vg;
        void assemble(Vec rt, int lev, double scale);
        void assemble_quad(Vec rt1, Vec rt2, int lev, double scale);
};

class EoSmat {
    public:
        EoSmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~EoSmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        double** Wt;
        double** WtQ;
        double** WtQW;
        double** Qaa;
        double* WtQWflat;
        M2_j_xy_i* W;
        Wii* Q;
        Mat M;
        void assemble(Vec rt, int lev, double scale);
};

class WmatInv {
    public:
        WmatInv(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~WmatInv();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale);
};

class WhmatInv {
    public:
        WhmatInv(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~WhmatInv();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec rho, int lev, double scale);
};

class N_rt_Inv {
    public:
        N_rt_Inv(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~N_rt_Inv();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(Vec rho, int lev, double scale, bool do_inverse);
};

class PtQUt_mat {
    public:
        PtQUt_mat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~PtQUt_mat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        double* QUflat;
        double* QVflat;
        double** Qaa;
        double** Qab;
        double** QU;
        double** QV;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        Wii* Q;
        void assemble(Vec u1, int lev, double scale);
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
        double* QUflat;
        double* QVflat;
        double** Qaa;
        double** Qab;
        double** QU;
        double** QV;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        Wii* Q;
        void assemble(Vec u1, int lev, double scale);
};

class WtQPmat {
    public:
        WtQPmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~WtQPmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale);
};

class N_RTmat {
    public:
        N_RTmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~N_RTmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale, Vec rt, Vec pi);
};

class N_PiInv_mat {
    public:
        N_PiInv_mat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~N_PiInv_mat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale, Vec rt, Vec pi);
};

class N_RT2_mat {
    public:
        N_RT2_mat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~N_RT2_mat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale, Vec rt);
};
