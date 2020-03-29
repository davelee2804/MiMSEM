class Solve3D {
    public:
        Solve3D(Topo* _topo, Geom* _geom, double dt, double del2);
        ~Solve3D();
        int rank;
        Topo* topo;
        Geom* geom;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        Umat* M1;
        Vec* ug;
        Vec ul;
        Vec vl;
        Vec x;
        Vec b;
        Mat M;
        KSP ksp;
        VecScatter scat;
        void RepackVector(Vec* vx, Vec v);
        void UnpackVector(Vec v, Vec* vx);
        void Solve(Vec* xg, Vec* bg, bool mult_rhs);

    private:
        int elOrd;
        int nElsX;
        int elOrd2;
        int* elInds2_l(int ex, int ey);
        int* elInds2_g(int ex, int ey);
};
