class VertOps {
    public:
        VertOps(Topo* _topo, Geom* _geom);
        ~VertOps();

        int   n2;
        Topo* topo;
        Geom* geom;

        GaussLobatto* quad;
        LagrangeNode* node;
        LagrangeEdge* edge;

        Wii* Q;
        M2_j_xy_i* W;
        double** Q0;
        double** QT;
        double** QB;
        double** Wt;
        double** WtQ;
        double** WtQW;
        double** WtQWinv;
        double* WtQWflat;

        Mat V01; // vertical divergence operator
        Mat V10; // vertical gradient operator
        Mat VA;
        Mat VB;
        Mat VA_inv;
        Mat VB_inv;
        Mat VAB;
        Mat VBA;
        Mat VR;  // rayleigh friction operator
        Mat VAB_w;

        void vertOps();

        void AssembleConst(int ex, int ey, Mat A);      // piecewise constant (in vertical) mass matrix
        void AssembleLinear(int ex, int ey, Mat B);     // piecewise linear (in vertical) mass matrix
        void AssembleLinCon(int ex, int ey, Mat AB);
        void AssembleLinearWithTheta(int ex, int ey, Vec theta, Mat A);
        void AssembleLinearWithRho(int ex, int ey, Vec* rho, Mat A, bool do_internal);
        void AssembleLinearWithRT(int ex, int ey, Vec rt, Mat A, bool do_internal);
        void AssembleLinearInv(int ex, int ey, Mat A);
        void AssembleConstWithRhoInv(int ex, int ey, Vec theta, Mat B);
        void AssembleConstWithRho(int ex, int ey, Vec rho, Mat A);
        void AssembleConLinWithW(int ex, int ey, Vec velz, Mat BA);
        void AssembleRayleigh(int ex, int ey, Mat B);
        void AssembleConstInv(int ex, int ey, Mat B);
        void Assemble_EOS_RHS(int ex, int ey, Vec rt, Vec eos_rhs);
        void AssembleLinConWithTheta(int ex, int ey, Mat AB, Vec theta);
};
