class Schur {
    public:
        Schur(Topo* _topo, Geom* _geom);
        ~Schur();
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
};
