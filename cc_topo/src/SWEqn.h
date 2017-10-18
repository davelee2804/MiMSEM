class SWEqn {
    public:
        SWEqn(Topo* _topo, Geom* _geom);
        ~SWEqn();
        double grav;
        double omega;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Topo* topo;
        Geom* geom;
        Pvec* m0;
        Umat* M1;
        Wmat* M2;
        E10mat* NtoE;
        E21mat* EtoF;
        RotMat* R;
        Uhmat* F;
        WtQUmat* K;
        Vec fl;            // coriolis vector (local)
        Vec fg;            // coriolis vector (global)
        Mat E01M1;
        Mat E12M2;
        VecScatter gtol_0;
        VecScatter gtol_1;
        VecScatter gtol_2;
        void coriolis();
        void diagnose_w(Vec u, Vec* w);
        void diagnose_F(Vec u, Vec h, Vec* hu);
        void solve(Vec ui, Vec hi, Vec uf, Vec hf, double dt);
};
