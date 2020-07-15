class Schur {
    public:
        Schur(Topo* _topo, Geom* _geom);
        ~Schur();
        int rank;
        Topo* topo;
        Geom* geom;
        Vec vl; // local vector in L2
        Vec x;
        Vec b;
        Vec t;
        Mat P; // [rho,rho] block
        Mat G; // [rho, rt] block
        Mat D; // [rt ,rho] block
        Mat T; // [rt , rt] block
        Mat Pinv;
        Mat DPinv;
        Mat S; // [T - DP^{-1}G]
        Mat Uinv;
        Mat Q;
        Mat D_rho;
        Mat D_rt;
        Mat DUinv;
        Mat L_rho;
        Mat L_rt;
        Mat A;
        Mat VISC;
        Mat Q2;
        Mat QT;
        Mat Minv;
        Mat QTMinv;
        Mat QTMinvQ;
        KSP ksp_rt;
        KSP ksp_rho;
        VecScatter scat;
        void AddFromVertMat(int ei, Mat Az, Mat _M);
        void AddFromHorizMat(int kk, Mat Ax, Mat _M);
        void RepackFromVert(Vec* vz, Vec v);
        void RepackFromHoriz(Vec* vx, Vec v);
        void UnpackToHoriz(Vec v, Vec* vx);
        void Solve(KSP ksp, Mat _M, L2Vecs* d_exner);
        void InitialiseMatrix();
        void DestroyMatrix();
        void AddFromVertMat_D(int ei, Mat Az, Mat _M);
        void AddFromHorizMat_D(int ei, Mat Az, Mat _M);
        void AddFromVertMat_G(int ei, Mat Az, Mat _M);
        void AddFromHorizMat_G(int ei, Mat Az, Mat _M);
        void AddFromVertMat_U(int ei, Mat Az, Mat _M);
        void AddFromHorizMat_U(int ei, Mat Az, Mat _M);
        void RepackFromVert_U(Vec* uz, Vec _u);
        void RepackFromHoriz_U(Vec* uz, Vec _u);

    private:
        int elOrd;
        int nElsX;
        int elOrd2;
        int* elInds2_l(int ex, int ey);
        int* elInds2_g(int ex, int ey);
};
