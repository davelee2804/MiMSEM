class Test {
    public:
        Test(SWEqn* _sw);
        ~Test();
        SWEqn* sw;
        void edgeFunc();
        void vorticity(ICfunc* fu, ICfunc* fv);
        void gradient(ICfunc* fh);
        void convection(ICfunc* fu, ICfunc* fv);
        void massFlux(ICfunc* fu, ICfunc* fv, ICfunc* fh);
        void kineticEnergy(ICfunc* fu, ICfunc* fv);
};
