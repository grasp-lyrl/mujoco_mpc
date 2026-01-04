#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"


namespace mjpc {

int Quadruped::ResidualFn::UprightCost(const mjData* data, 
                                       double* residual, int counter) const {

    double* torso_xmat = data->xmat + 9 * torso_body_id_;

    if (current_mode_ == kModeBiped) {
        double biped_type = parameters_[biped_type_param_id_];
        int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
        residual[counter++] = torso_xmat[6] - handstand;
    } else {
        residual[counter++] = torso_xmat[8] - 1;
    }

    residual[counter++] = 0;
    residual[counter++] = 0;
    return counter;
}

}  // namespace mjpc

