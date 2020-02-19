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
        Mat Q;
        Mat L;
        Mat L_rt;
        KSP ksp;
        VecScatter scat;
        void AddFromVertMat(int ei, Mat Az, Mat _M);
        void AddFromHorizMat(int kk, Mat Ax, Mat _M);
        void RepackFromVert(Vec* vz, Vec v);
        void RepackFromHoriz(Vec* vx, Vec v);
        void UnpackToHoriz(Vec v, Vec* vx);
        void Solve(L2Vecs* d_exner);
        void InitialiseMatrix();
        void DestroyMatrix();
        //void Preallocate(HorizSolve* hs, VertSolve* vs, L1Vecs* velx, L2Vecs* velz, L2Vecs* rho, L2Vecs* rt, L2Vecs* pi, 
        //                 L2Vecs* theta, L1Vecs* F_u, L2Vecs* F_w, L2Vecs* F_rho, L2Vecs* F_rt, L2Vecs* F_pi, L1Vecs* gradPi);

    private:
        int elOrd;
        int nElsX;
        int* inds2;
        int* elInds2_l(int ex, int ey);
        int* elInds2_g(int ex, int ey);
};
