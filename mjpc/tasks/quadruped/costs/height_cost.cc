#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>


namespace mjpc {

int Quadruped::ResidualFn::HeightCost(const mjData* data,
                                      const double torso_pos[3],
                                      double height_goal, // is_biped ? kHeightBiped : kHeightQuadruped
                                      const double avg_foot_pos[3],
                                      double* residual,
                                      int counter) const {

  if (current_mode_ == kModeScramble) {  // disable height term in Scramble
    residual[counter++] = 0;
  } else {
    residual[counter++] = (torso_pos[2] - avg_foot_pos[2]) - height_goal;
  }
  return counter;
}

}  // namespace mjpc
