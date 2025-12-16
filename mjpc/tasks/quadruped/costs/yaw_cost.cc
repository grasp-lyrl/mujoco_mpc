#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"


namespace mjpc {

int Quadruped::ResidualFn::YawCost(const mjModel* model, const mjData* data,
                                   const double* torso_xmat,
                                   double* residual, int counter) const {

    double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};

    if (current_mode_ == kModeBiped) {
        int handstand =
            ReinterpretAsInt(parameters_[biped_type_param_id_]) ? 1 : -1;
        torso_heading[0] = handstand * torso_xmat[2];
        torso_heading[1] = handstand * torso_xmat[5];
    }

    mju_normalize(torso_heading, 2);
    double heading_goal = parameters_[ParameterIndex(model, "Heading")];
    
    residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
    residual[counter++] = torso_heading[1] - mju_sin(heading_goal);

    return counter;
}

}  // namespace mjpc

