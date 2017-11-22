class Test {
    public:
        Test(SWEqn* _sw);
        ~Test();
        SWEqn* sw;
        void ke(ICfunc* fu, ICfunc* fv);
};
