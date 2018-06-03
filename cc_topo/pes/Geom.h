typedef double (TopogFunc) (double* xi);
typedef double (LevelFunc) (double* xi, int ki);

class Geom {
    public:
        Geom(int _pi, Topo* _topo, int _nk);
        ~Geom();
        int pi;
        int nl;
        int nk;        // number of vertical levels
        double** x;
        double** s;
        double** det;
        double**** J;
        double* topog;  // topography
        double** levs;  // vertical levels at each quadrature point
        double** thick; // layer thickness
        Topo* topo;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void interp0(int ex, int ey, int px, int py, double* vec, double* val);
        void interp1_l(int ex, int ey, int px, int py, double* vec, double* val);
        void interp2_l(int ex, int ey, int px, int py, double* vec, double* val);
        void interp1_g(int ex, int ey, int px, int py, double* vec, double* val);
        void interp2_g(int ex, int ey, int px, int py, double* vec, double* val);
        void write0(Vec q, char* fieldname, int tstep);
        void write1(Vec u, char* fieldname, int tstep);
        void write2(Vec h, char* fieldname, int tstep);
        void initJacobians();
        void updateGlobalCoords();
        void initTopog(TopogFunc* ft, LevelFunc* fl);
    private:
        void jacobian(int ex, int ey, int qx, int qy, double** J);
        double jacDet(int ex, int ey, int qx, int qy, double** J);
};
