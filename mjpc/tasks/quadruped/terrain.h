#pragma once

#include <vector>

#include <mujoco/mujoco.h>


namespace mjpc {

  class Terrain {

    public:
        Terrain() = default;
        explicit Terrain(const mjModel* model);

        void Initialize(const mjModel* model);

        void GetHeightFromLocal(double x, double y, double& z) const;
        void GetNormalFromLocal(double x, double y, double n[3]) const;

        void GetHeightFromWorld(const mjData* data, double x, double y, double& z) const;
        void GetNormalFromWorld(const mjData* data, double x, double y, double n[3]) const;

        int geom_id = -1;
        int hfield_id = -1;
        
        int nrow = 0, ncol = 0;
        double sx = 0, sy = 0, sz = 0;  // full hfield half sizes in m
        double dx = 0, dy = 0;  // cell sizes

    private:
        
        int adr = 0;
        const float* H = nullptr;  // pointer to hfield data

        struct BilinearParams {
            int x0, x1, y0, y1;
            double tx, ty;
        };

    public:
        BilinearParams LocalToGrid(double x, double y) const;
        void WorldToLocal(const mjData* data, double x, double y,
                          const double*& R, const double*& t,
                          double p_local[3]) const;

        struct PatchFeatures {
            double centroid[3];
            double normal[3];
            double roughness;
            double step_height;
            double max_height;
        };

        void GetPatchFeatures(const mjData* data, double x, double y,
                              PatchFeatures& features,
                              double patch_radius = 0.08) const;
        bool IsSafe(const mjData* data, double x, double y) const;

    private:
        static constexpr double kMaxRoughness = 0.02;
        static constexpr double kMinNormalZ = 0.7;
        static constexpr double kMaxStepHeight = 0.05;

    void GetVertexNormal(int r, int c, float* n) const;
  };

}

