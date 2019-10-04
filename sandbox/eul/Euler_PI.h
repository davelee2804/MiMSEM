typedef double (ICfunc3D) (double* xi, int ki);

class Euler {
    public:
        Euler(Topo* _topo, Geom* _geom, double _dt);
        ~Euler();
        double dt;
        int rank;
        int step;
        bool firstStep;
        Topo* topo;
        Geom* geom;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Schur* schur;
        VertSolve* vert;
        HorizSolve* horiz;

        void dump(Vec* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* theta, int num);

        void solve(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);

        void init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* h, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);
};
