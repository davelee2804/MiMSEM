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
        void jacobian(int ex, int ey, int qx, int qy, double** J);
        double jacDet(int ex, int ey, int qx, int qy, double** J);
        void Write0(Vec q, char* fieldname, int tstep);
        void Write1(Vec u, char* fieldname, int tstep);
        void Write2(Vec h, char* fieldname, int tstep);
};
