typedef double (ICfunc3D) (double* xi, int ki);

class HorizSolve {
    public:
        HorizSolve(Topo* _topo, Geom* _geom);
        ~HorizSolve();
        double del2;
        double k2i;
        bool do_visc;
        bool do_temp_visc;
        int rank;
        int size;
        int step;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        VertOps* vo;
        Pmat* M0;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* F;
        WtQUmat* K;
        Ut_mat* M1t;
        UtQWmat* Rh;
        Whmat* M2h_mat;
        Uvec* m1;
        Wvec* m2;
        Vec* fg;                 // coriolis vector (global)
        Vec* fl;                 // coriolis vector (local)
        Vec* Fk;
        Vec* Gk;
        KSP ksp1;
        KSP ksp0;

        double viscosity();
        void coriolis();
        void grad(bool assemble, Vec phi, Vec* u, int lev); // weak form grad operator
        void curl(bool assemble, Vec u, Vec* w, int lev, bool add_f);      // weak form curl operator
        void laplacian(bool assemble, Vec u, Vec* ddu, int lev);       // laplacian operator via helmholtz decomposition

        void diagnose_fluxes(int level, Vec u1, Vec u2, Vec h1l, Vec h2l, Vec* theta_l, Vec _F, Vec _G, Vec u1l, Vec u2l, bool theta_in_Wt);
        void advection_rhs(Vec* u1, Vec* u2, Vec* h1l, Vec* h2l, L2Vecs* theta, L2Vecs* dF, L2Vecs* dG, Vec* u1l, Vec* u2l);
        void advection_rhs_ec(Vec* u1, Vec* u2, Vec* h1l, Vec* h2l, Vec* theta, L2Vecs* dF, L2Vecs* dG, Vec* u1l, Vec* u2l);

        void diagnose_Phi(int level, Vec u1, Vec u2, Vec u1l, Vec u2l, Vec* velz1, Vec* velz2, Vec* Phi);
        void diagnose_q(int level, Vec rho, Vec ul, Vec* qi);
        void diagHorizVort(Vec* velx, Vec* dudz);
        void diagVertVort(Vec* velz, Vec* rho, Vec* dwdx);

        void momentum_rhs(int level, Vec* theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                          Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec rho1, Vec rho2, Vec fu, Vec Fx, Vec* Fz, Vec* dwdx1, Vec* dwdx2);
        void momentum_rhs_ec(int level, Vec theta, Vec* dudz1, Vec* dudz2, Vec* velz1, Vec* velz2, Vec Pi, 
                          Vec velx1, Vec velx2, Vec uil, Vec ujl, Vec rho1, Vec rho2, Vec fu, Vec Fx, Vec* Fz, Vec* dwdx1, Vec* dwdx2);
};
