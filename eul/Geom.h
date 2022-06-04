typedef double (TopogFunc) (double* xi);
typedef double (LevelFunc) (double* xi, int ki);

class Geom {
    public:
        Geom(Topo* _topo, int _nk);
        ~Geom();
        int pi;
        int nl;
        int nk;        // number of vertical levels
	int nDofsX;
	int nDofs0G;
	int n0;
	int n0l;
	int* loc0;
	int* inds0_l;
	int* inds0_g;
	IS is_l_0;
	IS is_g_0;
	VecScatter gtol_0;
        double** x;
        double** s;
        double** det;
        double**** J;
        double* topog;  // topography
        double** levs;  // vertical levels at each quadrature point
        double** thick; // layer thickness
        double** thickInv; // layer thickness inverse
        double* WA;
        double* UA;
        double* VA;
        Topo* topo;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void interp0(int ex, int ey, int px, int py, double* vec, double* val);
        void interp1_l(int ex, int ey, int px, int py, double* vec, double* val);
        void interp2_l(int ex, int ey, int px, int py, double* vec, double* val);
        void interp1_g(int ex, int ey, int px, int py, double* vec, double* val);
        void interp1_g_t(int ex, int ey, int px, int py, double* vec, double* val);
        void interp2_g(int ex, int ey, int px, int py, double* vec, double* val);
        void write0(Vec q, char* fieldname, int tstep, int kk);
        void write1(Vec u, char* fieldname, int tstep, int kk);
        void write2(Vec h, char* fieldname, int tstep, int kk, bool vert_scale);
        void writeVertToHoriz(Vec* vecs, char* fieldname, int tstep, int nv);
        void initJacobians();
        void updateGlobalCoords();
        void initTopog(TopogFunc* ft, LevelFunc* fl);
        void writeColumn(char* filename, int ei, int nv, Vec vecs, bool vert_scale);
        int* elInds0_l(int ex, int ey);
        int* elInds0_g(int ex, int ey);
    private:
        void jacobian(int ex, int ey, int qx, int qy, double** J);
        double jacDet(int ex, int ey, int qx, int qy, double** J);
};
