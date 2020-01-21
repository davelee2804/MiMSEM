class Schur {
    public:
        Schur(Topo* _topo, Geom* _geom);
        ~Schur();
        int rank;
        Topo* topo;
        Geom* geom;
        Vec vl;
        Vec x;
        Vec b;
        Mat M;
        KSP ksp;
        VecScatter scat;
        void AddFromVertMat(int ei, Mat Az);
        void AddFromHorizMat(int kk, Mat Ax);
        void RepackFromVert(Vec* vz, Vec v);
        void RepackFromHoriz(Vec* vx, Vec v);
        void UnpackToHoriz(Vec v, Vec* vx);
        void Solve(L2Vecs* d_exner);
        void InitialiseMatrix();
        void DestroyMatrix();

    private:
        int elOrd;
        int nElsX;
        int* inds2;
        int* elInds2_l(int ex, int ey);
        int* elInds2_g(int ex, int ey);
};
