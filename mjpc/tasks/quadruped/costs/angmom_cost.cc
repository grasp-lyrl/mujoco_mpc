#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"


namespace mjpc {

int Quadruped::ResidualFn::AngularMomentumCost(const mjModel* model,
                                               const mjData* data,
                                               double* residual,
                                               int counter) const {

    mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));

    counter += 3;
    return counter;
}

}  // namespace mjpc

