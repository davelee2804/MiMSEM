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
        void assemble(Vec* q0, double scale, Mat M);
};
