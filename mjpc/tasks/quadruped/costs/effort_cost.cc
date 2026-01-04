#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>


namespace mjpc {

int Quadruped::ResidualFn::EffortCost(const mjModel* model,
                                      const mjData* data,
                                      double* residual,
                                      int counter) const {

    mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
    counter += model->nu;
    
    return counter;
}

}  // namespace mjpc

