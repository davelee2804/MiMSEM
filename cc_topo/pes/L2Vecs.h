class L2Vecs {
    public:
        L2Vecs(int _nk, Topo* _topo, Geom* _geom);
        ~L2Vecs();
        int nk;
        Topo* topo;
        Geom* geom;
        Vec* vh;
        Vec* vz;
        Vec* vl;
        void VertToHoriz();
        void HorizToVert();
        void UpdateLocal();
        void UpdateGlobal();
        void CopyFromVert(Vec* vf);
        void CopyFromHoriz(Vec* vf);
        void CopyToVert(Vec* vf);
        void CopyToHoriz(Vec* vf);
};
