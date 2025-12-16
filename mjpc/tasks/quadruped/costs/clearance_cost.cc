#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"
#include "mjpc/tasks/quadruped/terrain.h"


namespace mjpc {

int Quadruped::ResidualFn::ClearanceCost(const mjModel* model,
                                         const mjData* data,
                                         double* residual,
                                         int counter) const {

    const double r = box_half_size_[0];
    const double offsets[5][2] = {{0, 0}, {r, 0}, {-r, 0}, {0, r}, {0, -r}};
    constexpr double kMargin = 0.03;        // in [m]
    constexpr double kBeta = 200.0;         
    constexpr double kNegInf = -1e6;        
    double z;

    for (A1Foot foot : kFootAll) {
        const int body_ids[2] = {knee_body_id_[foot], shoulder_body_id_[foot]};

        for (int i = 0; i < 2; ++i) {
            const double* pos = data->xpos + 3 * body_ids[i];
            double z_safe = kNegInf;
            for (const auto& off : offsets) {
                terrain_->GetHeightFromWorld(data, pos[0] + off[0], pos[1] + off[1], z);
                z_safe = mju_max(z_safe, z);
            }
            double gap = pos[2] - z_safe;
            double penalty = std::log1p(mju_exp(kBeta * (kMargin - gap))) / kBeta;
            residual[counter++] = penalty;
        }
    }

    // lidar
    const double* pos = data->geom_xpos + 3 * lidar_geom_id_;
    double z_safe = kNegInf;
    for (const auto& off : offsets) {
        terrain_->GetHeightFromWorld(data, pos[0] + off[0], pos[1] + off[1], z);
        z_safe = mju_max(z_safe, z);
    }
    double gap = pos[2] - z_safe;
    double penalty = std::log1p(mju_exp(kBeta * (kMargin - gap))) / kBeta;
    residual[counter++] = penalty;

  return counter;
}

}  // namespace mjpc
