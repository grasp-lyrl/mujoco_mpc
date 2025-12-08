#pragma once

#include <vector>

#include <mujoco/mujoco.h>


namespace mjpc {

  class Terrain {

    public:
        Terrain() = default;
        explicit Terrain(const mjModel* model);

        void Initialize(const mjModel* model);
        bool UpdateNormals();

        void GetHeightFromLocal(double x, double y, double& z) const;
        void GetNormalFromLocal(double x, double y, double n[3]) const;

        bool GetHeightFromWorld(const mjData* data, double x, double y, double& z) const;
        bool GetNormalFromWorld(const mjData* data, double x, double y, double n[3]) const;

        int geom_id = -1;
        int hfield_id = -1;
        
        int nrow = 0, ncol = 0;
        double sx = 0, sy = 0, sz = 0;  // full hfield half sizes in m
        double dx = 0, dy = 0;  // cell sizes

    private:
        
        int adr = 0;
        const float* H = nullptr;  // pointer to hfield data
        std::vector<float> normals;

        struct BilinearParams {
            int x0, x1, y0, y1;
            double tx, ty;
        };

    BilinearParams CoordsLocalToGrid(double x, double y) const;
  };

}

