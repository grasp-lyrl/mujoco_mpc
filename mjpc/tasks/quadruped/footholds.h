#ifndef MJPC_TASKS_QUADRUPED_FOOTHOLDS_H_
#define MJPC_TASKS_QUADRUPED_FOOTHOLDS_H_

#include <mujoco/mujoco.h>
#include "mjpc/tasks/quadruped/quadruped.h"


namespace mjpc {

void ComputeFootholds(const mjModel* model, mjData* data,
                      const mjpc::Quadruped::ResidualFn& residual,
                      double duty_ratio);


void ComputeFootholdTarget(const mjModel* model, const mjData* data,
                           const mjpc::Quadruped::ResidualFn& residual,
                           mjpc::Quadruped::ResidualFn::A1Foot foot,
                           mjpc::Quadruped::ResidualFn::A1Gait gait,
                           double phase, double duty_ratio, double step_height,
                           const double torso_x[3], double out_target[3]);


double SwingPhase(double phase, double footphase, double duty_ratio);

bool GetLatchedControlPoints(mjpc::Quadruped::ResidualFn::A1Foot foot,
                             double ctrl_pts[4][3]);
bool IsFootSwinging(mjpc::Quadruped::ResidualFn::A1Foot foot);
bool IsBezierActive(mjpc::Quadruped::ResidualFn::A1Foot foot);
void EvalLatchedBezier(mjpc::Quadruped::ResidualFn::A1Foot foot, double t,
                       double out[3]);

}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_FOOTHOLDS_H_

