typedef double (ICfunc3D) (double* xi, int ki);

class Euler {
    public:
        Euler(Topo* _topo, Geom* _geom, double _dt);
        ~Euler();
        double dt;
        int rank;
        int step;
        Topo* topo;
        Geom* geom;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        VertSolve* vert;
        HorizSolve* horiz;
        Schur* schur;
        Schur* schur_rho;

        void dump(Vec* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* exner, L2Vecs* theta, int num);

        void solve_schur(Vec* velx_i, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);
        void solve_gauss_seidel(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);
        void solve_schur_2(Vec* velx, L2Vecs* velz_i, L2Vecs* rho_i, L2Vecs* rt_i, L2Vecs* exner_i, bool save);

        void init1(Vec *u, ICfunc3D* func_x, ICfunc3D* func_y);
        void init2(Vec* h, ICfunc3D* func);
        void initTheta(Vec theta, ICfunc3D* func);

        void GlobalNorms(int itt, Vec* duh, Vec* uh, L2Vecs* duz, L2Vecs* uz, L2Vecs* drho, L2Vecs* rho, L2Vecs* drt, L2Vecs* rt, L2Vecs* dexner, L2Vecs* exner,
                         double* norm_u, double* norm_w, double* norm_rho, double* norm_rt, double* norm_exner, Vec h_tmp, Vec u_tmp, Vec u_tmp_z, bool prnt);

        double ComputeAlpha(Vec* velz_i, Vec* velz_j, Vec* d_velz, 
                            Vec* rho_i, Vec* rho_j, Vec* d_rho, 
                            Vec* rt_i, Vec* rt_j, Vec* d_rt, 
                            Vec* pi_i, Vec* pi_j, Vec* d_pi, Vec* pi_h,
                            Vec* theta_i, Vec* theta_h);
};
