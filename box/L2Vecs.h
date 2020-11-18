class L2Vecs {
    public:
        L2Vecs(int _nk, Topo* _topo, Geom* _geom);
        ~L2Vecs();
        int nk;
        Topo* topo;
        Geom* geom;
        Vec* vh;
        Vec* vz;
        void VertToHoriz();
        void HorizToVert();
        void CopyFromVert(Vec* vf);
        void CopyFromHoriz(Vec* vf);
        void CopyToVert(Vec* vf);
        void CopyToHoriz(Vec* vf);
};

class L1Vecs {
    public:
        L1Vecs(int _nk, Topo* _topo, Geom* _geom);
        ~L1Vecs();
        int nk;
        Topo* topo;
        Geom* geom;
        Vec* vh;
        Vec* vl;
        void UpdateLocal();
        void UpdateGlobal();
        void CopyFrom(Vec* vf);
        void CopyTo(Vec* vf);
};
