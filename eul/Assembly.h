class Umat {
    public:
        Umat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        Mat MT;
        Mat _M;
        void assemble(int lev, double scale, bool vert_scale);
        void assemble_up(int lev, double scale, double tau, Vec ui, Vec uj);
        void _assemble(int lev, double scale, bool vert_scale);
};

class Wmat {
    public:
        Wmat(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~Wmat();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        Mat M;
        Mat _M;
        void assemble(int lev, double scale, bool vert_scale);
        void _assemble(int lev, double scale, bool vert_scale);
};

class Uhmat {
    public:
        Uhmat(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Uhmat();
        double* UtQUflat;
        double* UtQU;
        double* UtQV;
        double* VtQU;
        double* VtQV;
        double* Qaa;
        double* Qab;
        double* Qbb;
        double* Ut;
        double* Vt;
        double* UtQaa;
        double* UtQab;
        double* VtQba;
        double* VtQbb;
        Wii* Q;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        Mat MT;
        void assemble(Vec h2, int lev, bool const_vert, double scale);
        void assemble_up(Vec h2, int lev, double scale, double dt, Vec u1);
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
        double* Wt;
        double* Qaa;
        double* Qab;
        double* WtQaa;
        double* WtQab;
        double* WtQU;
        double* WtQV;
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
        double* Ut;
        double* Vt;
        double* Qab;
        double* Qba;
        double* UtQab;
        double* VtQba;
        double* UtQV;
        double* VtQU;
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
        Wii* Q;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        double* Ut;
        double* Vt;
        double* UtQaa;
        double* UtQab;
        double* VtQba;
        double* VtQbb;
        double* UtQU;
        double* UtQV;
        double* VtQU;
        double* VtQV;
        double* Qaa;
        double* Qab;
        double* Qbb;
        double* UtQUflat;
        Mat M;
        void assemble(int lev, double scale);
        void assemble_h(int lev, double scale, Vec rho);
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
        double* Ut;
        double* Vt;
        double* Qaa;
        double* Qba;
        double* UtQaa;
        double* VtQba;
        double* UtQW;
        double* VtQW;
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
        double* Wt;
        double* Qaa;
        double* Qab;
        double* WtQaa;
        double* WtQab;
        double* WtQU;
        double* WtQV;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        M2_j_xy_i* W;
        Wii* Q;
        void assemble(Vec u1, double scale);
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
        double* Qaa;
        double* Qab;
        double* QU;
        double* QV;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        Wii* Q;
        void assemble(Vec u1, int lev, double scale);
};

class Umat_ray {
    public:
        Umat_ray(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat_ray();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        void assemble(int lev, double scale, double dt, Vec exner, Vec exner_s);
};

//////////////////////////////////////////////////////////////////
class Uvec {
    public:
        Uvec(Topo* _topo, Geom* _geom, LagrangeNode* _node, LagrangeEdge* _edge);
        ~Uvec();
        Topo* topo;
        Geom* geom;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Vec vl;
        Vec vg;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        Wii* Q;
        double* Ut;
        double* Vt;
        void assemble(int lev, double scale, bool vert_scale, Vec vel);
        void assemble_hu(int lev, double scale, Vec vel, Vec rho, bool zero_and_scatter, double fac);
        void assemble_wxu(int lev, double scale, Vec vel, Vec vort);
        void assemble_hu_up(int lev, double scale, Vec vel, Vec rho, double fac, double tau, Vec vel2);
};

class Wvec {
    public:
        Wvec(Topo* _topo, Geom* _geom, LagrangeEdge* _edge);
        ~Wvec();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* edge;
        Vec vg;
        M2_j_xy_i* W;
        Wii* Q;
        double* Wt;
        void assemble(int lev, double scale, bool vert_scale, Vec rho);
        void assemble_K(int lev, double scale, Vec vel1, Vec vel2);
};
