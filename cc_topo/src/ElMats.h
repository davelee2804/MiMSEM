double** Mult(int ni, int nj, int nk, double** A, double** B);
void Mult_IP(int ni, int nj, int nk, double** A, double** B, double** C);
void MultVec_IP(int ni, int nj, int nk, double** A1, double** B1, double** A2, double** B2, double** C);
double** Tran(int ni, int nj, double** A);
void Tran_IP(int ni, int nj, double** A, double** B);

class M1x_j_xy_i {
    public:
        M1x_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1x_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
};

class M1y_j_xy_i {
    public:
        M1y_j_xy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1y_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
};

class M1x_j_Cxy_i {
    public:
        M1x_j_Cxy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1x_j_Cxy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M1y_j_Cxy_i {
    public:
        M1y_j_Cxy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1y_j_Cxy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M1x_j_Exy_i {
    public:
        M1x_j_Exy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1x_j_Exy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M1y_j_Exy_i {
    public:
        M1y_j_Exy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1y_j_Exy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M1x_j_Dxy_i {
    public:
        M1x_j_Dxy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1x_j_Dxy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M1y_j_Dxy_i {
    public:
        M1y_j_Dxy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1y_j_Dxy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M1x_j_Fxy_i {
    public:
        M1x_j_Fxy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1x_j_Fxy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M1y_j_Fxy_i {
    public:
        M1y_j_Fxy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M1y_j_Fxy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
};

class M2_j_xy_i {
    public:
        M2_j_xy_i(LagrangeEdge* _edge);
        ~M2_j_xy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeEdge* edge;
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

class M0_j_Cxy_i {
    public:
        M0_j_Cxy_i(LagrangeNode* _node, LagrangeEdge* _edge);
        ~M0_j_Cxy_i();
        int nDofsI;
        int nDofsJ;
        double** A;
        LagrangeNode* node;
        LagrangeEdge* edge;
        void assemble(double* c);
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
        void assemble(int ex, int ey);
};

class JacM2 {
    public:
        JacM2(GaussLobatto* _quad, Geom* _geom);
        ~JacM2();
        int nDofsI;
        int nDofsJ;
        double** J;
        double** A;
        GaussLobatto* quad;
        Geom* geom;
        void assemble(int ex, int ey);
};

class JacM1 {
    public:
        JacM1(GaussLobatto* _quad, Geom* _geom);
        ~JacM1();
        int nDofsI;
        int nDofsJ;
        double** J;
        double** Aaa;
        double** Aab;
        double** Aba;
        double** Abb;
        GaussLobatto* quad;
        Geom* geom;
        void assemble(int ex, int ey);
};
