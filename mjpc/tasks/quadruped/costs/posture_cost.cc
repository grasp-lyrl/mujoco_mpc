#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"


namespace mjpc {

int Quadruped::ResidualFn::PostureCost(const mjModel* model,
                                       const mjData* data, double* residual,
                                       int counter) const {

    double* home = KeyQPosByName(model, data, "home");
    mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);

    for (A1Foot foot : kFootAll) {
        for (int joint = 0; joint < 3; joint++) {
            residual[counter + 3 * foot + joint] *= kJointPostureGain[joint];
        }
    }

    if (current_mode_ == kModeBiped) {  // loosen the "hands" in Biped mode
        bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
        double arm_posture = parameters_[arm_posture_param_id_];

        if (handstand) {
            residual[counter + 6] *= arm_posture;
            residual[counter + 7] *= arm_posture;
            residual[counter + 8] *= arm_posture;
            residual[counter + 9] *= arm_posture;
            residual[counter + 10] *= arm_posture;
            residual[counter + 11] *= arm_posture;
        } else {
            residual[counter + 0] *= arm_posture;
            residual[counter + 1] *= arm_posture;
            residual[counter + 2] *= arm_posture;
            residual[counter + 3] *= arm_posture;
            residual[counter + 4] *= arm_posture;
            residual[counter + 5] *= arm_posture;
        }
    }
    
    counter += model->nu;
    return counter;
}

}  // namespace mjpc

