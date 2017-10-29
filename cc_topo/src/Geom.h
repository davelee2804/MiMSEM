class Geom {
    public:
        Geom(int _pi, Topo* _topo);
        ~Geom();
        int pi;
        int nl;
        double** x;
        double** s;
        Topo* topo;
        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;
        double** R;
        void jacobian(int ex, int ey, int qx, int qy, double** J);
        double jacDet(int ex, int ey, int qx, int qy, double** J);
        void write0(Vec q, char* fieldname, int tstep);
        void write1(Vec u, char* fieldname, int tstep);
        void write2(Vec h, char* fieldname, int tstep);
};
