class M1x_j_xy_i {
    public:
        M1x_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1x_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
	int n_coarse;
	double** A_coarse;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void eval_at_pts(double** pts, int np);
};

class M1y_j_xy_i {
    public:
        M1y_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1y_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
	int n_coarse;
	double** A_coarse;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void eval_at_pts(double** pts, int np);
};

class M2_j_xy_i {
    public:
        M2_j_xy_i(LagrangeEdge* _edge);
        ~M2_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
	int n_coarse;
	double** A_coarse;
        LagrangeEdge* edge;
        void eval_at_pts(double** pts, int np);
};

class M0_j_xy_i {
    public:
        M0_j_xy_i(LagrangeNode* _node);
        ~M0_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
};

class Wii {
    public:
        Wii(GaussLobatto* _quad, Geom* _geom);
        ~Wii();
        int nDofsI;
        int nDofsJ;
        double** J;
        double** A;
        GaussLobatto* quad;
        Geom* geom;
    private:
        void assemble();
};
