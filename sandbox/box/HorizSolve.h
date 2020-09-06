typedef double (ICfunc3D) (double* xi, int ki);

class HorizSolve {
    public:
        HorizSolve(Topo* _topo, Geom* _geom, double _dt);
        ~HorizSolve();
        double dt;
        double del2;
        double k2i;
        bool do_visc;
        int rank;
        int size;
        int step;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        VertOps* vo;
        Pvec* m0;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* F;
        WtQUmat* K;
        Ut_mat* M1t;
        UtQWmat* Rh;
        Vec* Fk;
        KSP ksp1;
        KSP ksp1o;
        Uvec* M1u;
        Wvec* M2h;

        double viscosity();
        void grad(KSP _ksp, Mat _M2, Vec phi, Vec* u); // weak form grad operator
        void curl(Vec u, Vec* w);      // weak form curl operator
        void laplacian(Vec u, Vec* ddu);       // laplacian operator via helmholtz decomposition

        void diagnose_Pi(int level, Vec rt1, Vec rt2, Vec Pi);

        void diagnose_fluxes(int level, Vec u1, Vec u2, Vec h1l, Vec h2l, Vec* theta_l, Vec _F, Vec _G, Vec u1l, Vec u2l);
        void advection_rhs(Vec* u1, Vec* u2, Vec* h1l, Vec* h2l, L2Vecs* theta, L2Vecs* dF, L2Vecs* dG, Vec* u1l, Vec* u2l, bool do_temp_visc);

        void diagnose_Phi(int level, Vec u1, Vec u2, Vec u1l, Vec u2l, Vec* Phi);
        void diagnose_wxu(int level, Vec u1, Vec u2, Vec* wxu);
        void diagHorizVort(Vec* velx, Vec* dudz);

        void momentum_rhs(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                          Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec fu);
};
