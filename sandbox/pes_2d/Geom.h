typedef double (TopogFunc) (double xi);
typedef double (LevelFunc) (double xi, int ki);

class Geom {
    public:
        Geom(Topo* _topo, int _nk, double _lx);
        ~Geom();
        int nl;
        int nk;        // number of vertical levels
        double lx;
        double* x;
        double** det;
        double* topog;  // topography
        double** levs;  // vertical levels at each quadrature point
        double** thick; // layer thickness
        Topo* topo;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void interp0(int ex, int px, double* vec, double* val);
        void interp1(int ex, int px, double* vec, double* val);
        void interp2(int ex, int px, double* vec, double* val);
        void write0(Vec *q, char* fieldname, int tstep);
        void write1(Vec *u, char* fieldname, int tstep);
        void write2(Vec *h, char* fieldname, int tstep, bool const_vert);
        void writeSerial(Vec* vecs, char* fieldname, int tstep, int nv);
        void initJacobians();
        void initTopog(TopogFunc* ft, LevelFunc* fl);
    private:
        double jacDet(int ex, int qx);
};
