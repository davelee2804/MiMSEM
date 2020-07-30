typedef double (ICfunc) (double* xi);

class AdvEqn {
    public:
        AdvEqn(Topo* _topo, Geom* _geom);
        ~AdvEqn();
        double dt;
        int step;
        int rank;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        Umat* M1;
        Wmat* M2;
        WtQUmat* K;
        E21mat* EtoF;
        Mat KT;
        KSP ksp1;
        void init1(Vec u, ICfunc* func_x, ICfunc* func_y);
        void init2(Vec h, ICfunc* func);
        void err2(Vec u, ICfunc* fu, double* norms);
        void diagnose_F(Vec _u, Vec _q, Vec dF);
        void solve(Vec ui, Vec qi, Vec uj, Vec qj, double _dt, bool save);
};
