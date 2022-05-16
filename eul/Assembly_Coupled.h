class Umat_coupled {
    public:
        Umat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Umat_coupled();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        void assemble(double scale, Mat M);
};

class Wmat_coupled {
    public:
        Wmat_coupled(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~Wmat_coupled();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
        void assemble(double scale, int var_ind, Mat M);
};

class RotMat_coupled {
    public:
        RotMat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~RotMat_coupled();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
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
        void assemble(double scale, double fac, Vec* q0, Mat M);
};

class EoSmat_coupled {
    public:
        EoSmat_coupled(Topo* _topo, Geom* _geom, LagrangeEdge* _edge);
        ~EoSmat_coupled();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* edge;
        Wii* Q;
        M2_j_xy_i* W;
        double* Wt;
        double* WtQ;
        double* WtQW;
        double* WtQWinv;
        double* AAinv;
        double* AAinvA;
        void assemble(double scale, double fac, int col_ind, Vec* p2, Mat M);
        void assemble_rho_inv_mm(double scale, double fac, int lev, Vec p2, Mat M2);
};

void AddGradx_Coupled(Topo* topo, int lev, int var_ind, Mat G, Mat M);
void AddDivx_Coupled(Topo* topo, int lev, int var_ind, Mat G, Mat M);
void AddQx_Coupled(Topo* topo, int lev, Mat Q, Mat M);
void AddGradz_Coupled(Topo* topo, int ex, int ey, int var_ind, Mat G, Mat M);
void AddDivz_Coupled(Topo* topo, int ex, int ey, int var_ind, Mat D, Mat M);
void AddMz_Coupled(Topo* topo, int ex, int ey, int var_ind, Mat Mz, Mat M);
void AddQz_Coupled(Topo* topo, int ex, int ey, Mat Q, Mat M);

class E32_Coupled {
    public:
        E32_Coupled(Topo* _topo);
        ~E32_Coupled();
        Topo* topo;
        Mat M;
        Mat MT;
};

class M3mat_coupled {
    public:
        M3mat_coupled(Topo* _topo, Geom* _geom, LagrangeEdge* _e);
        ~M3mat_coupled();
        Topo* topo;
        Geom* geom;
        LagrangeEdge* e;
	Mat M;
        void assemble(double scale, Vec* p3, bool vert_scale, double fac);
};

class M2mat_coupled {
    public:
        M2mat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~M2mat_coupled();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
	Mat M;
        void assemble(double scale, double fac, Vec* p3);
};

class Kmat_coupled {
    public:
        Kmat_coupled(Topo* _topo, Geom* _geom, LagrangeNode* _l, LagrangeEdge* _e);
        ~Kmat_coupled();
        Topo* topo;
        Geom* geom;
        LagrangeNode* l;
        LagrangeEdge* e;
        Mat M;
        double* UtQWflat;
        double* VtQWflat;
        double* Ut;
        double* Vt;
        double* Wt;
        double* Qaa;
        double* Qba;
        double* UtQaa;
        double* VtQba;
        double* WtQ;
        double* UtQW;
        double* VtQW;
        double* WtQW;
        M1x_j_xy_i* U;
        M1y_j_xy_i* V;
        M2_j_xy_i* W;
        Wii* Q;
        void assemble(Vec* ul, Vec* wl, double fac, double scale);
};
