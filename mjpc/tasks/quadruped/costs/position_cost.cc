#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>


namespace mjpc {

int Quadruped::ResidualFn::PositionCost(const mjData* data, double* goal_pos,
                                        double* residual,
                                        int counter) const {

    double* head = data->site_xpos + 3 * head_site_id_;
    double target[3];

    if (current_mode_ == kModeWalk) { // follow prescribed Walk trajectory
        double mode_time = data->time - mode_start_time_;
        Walk(target, mode_time);
    } else { // go to the goal mocap body
        target[0] = goal_pos[0];
        target[1] = goal_pos[1];
        target[2] = goal_pos[2];
    }

    residual[counter++] = head[0] - target[0];
    residual[counter++] = head[1] - target[1];
    residual[counter++] =
    current_mode_ == kModeScramble ? 2 * (head[2] - target[2]) : 0;
    
    return counter;
}

}  // namespace mjpc

