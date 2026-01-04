#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"


namespace mjpc {

int Quadruped::ResidualFn::BalanceCost(const mjModel* model,
                                       const mjData* data, double height_goal,
                                       const double avg_foot_pos[3],
                                       double* compos, double* residual,
                                       int counter) const {

    double* comvel = SensorByName(model, data, "torso_subtreelinvel");

    double capture_point[3];
    double fall_time = mju_sqrt(2 * height_goal / 9.81);
    mju_addScl3(capture_point, compos, comvel, fall_time);

    residual[counter++] = capture_point[0] - avg_foot_pos[0];
    residual[counter++] = capture_point[1] - avg_foot_pos[1];

    return counter;
}

}  // namespace mjpc

