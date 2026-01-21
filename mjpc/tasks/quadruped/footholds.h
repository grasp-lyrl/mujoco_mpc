#ifndef MJPC_TASKS_QUADRUPED_FOOTHOLDS_H_
#define MJPC_TASKS_QUADRUPED_FOOTHOLDS_H_

#include <mujoco/mujoco.h>
#include "mjpc/tasks/quadruped/quadruped.h"


namespace mjpc {

class FootholdPlanner {
    public:
        static double SwingPhase(double phase, double footphase, double duty_ratio);
        static bool IsSwinging(double phase, double footphase, double duty_ratio);

        void ComputeFootholds(const mjModel* model, mjData* data,
                              const mjpc::Quadruped::ResidualFn& residual,
                              double duty_ratio);

        void EvalBezier(mjpc::Quadruped::ResidualFn::A1Foot foot,
                        double t, double out[3]) const;

        bool bezier_active_[mjpc::Quadruped::ResidualFn::kNumFoot] = {};
        double ctrl_pts_[mjpc::Quadruped::ResidualFn::kNumFoot][4][3] = {};

    private:
        bool in_swing_[mjpc::Quadruped::ResidualFn::kNumFoot] = {};
};

}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_FOOTHOLDS_H_

