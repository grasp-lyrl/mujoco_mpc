#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/quadruped/footholds.h"
#include "mjpc/utilities.h"

#include <mujoco/mujoco.h>

#include <cmath>


namespace mjpc {

int Quadruped::ResidualFn::GaitCost(const mjModel* model, const mjData* data,
                                    const double torso_pos[3], bool is_biped,
                                    double* foot_pos[kNumFoot],
                                    double* goal_pos, double* residual,
                                    int counter) const {

    A1Gait gait = GetGait();
    double step[kNumFoot];
    FootStep(step, GetPhase(data->time), gait);

    // Optional user-sensor foothold targets (expected 12 values: 3 per foot in FL, HL, FR, HR)
    double* foothold_targets = SensorByName(model, data, "foothold_targets");

    for (A1Foot foot : kFootAll) {
        if (is_biped) {  // ignore "hands" in biped mode

            bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
            bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
            bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);

            if (front_hand || back_hand) {
                residual[counter++] = 0.0;
                residual[counter++] = 0.0;
                residual[counter++] = 0.0;

                continue;
            }
        }

        const bool in_swing = IsFootSwinging(foot);
        const bool bezier_active = IsBezierActive(foot);

        if (foothold_targets && bezier_active) { // track Bezier: 3D in swing, XY-only in stance.
            double* target = foothold_targets + 3 * foot;
            residual[counter++] = foot_pos[foot][0] - target[0];
            residual[counter++] = foot_pos[foot][1] - target[1];
            residual[counter++] = in_swing ? (foot_pos[foot][2] - target[2]) : 0.0;
        } else if (in_swing) {  // swing but no active Bezier: height-only
            double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};
            if (current_mode_ == kModeScramble && torso_pos && goal_pos) {
                // Match the original Scramble behavior: sample ground 15cm toward goal.
                double torso_to_goal[3];
                mju_sub3(torso_to_goal, goal_pos, torso_pos);
                mju_normalize3(torso_to_goal);

                // Direction from current foot to goal, in XY only.
                mju_sub3(torso_to_goal, goal_pos, foot_pos[foot]);
                torso_to_goal[2] = 0.0;
                mju_normalize3(torso_to_goal);
                mju_addToScl3(query, torso_to_goal, 0.15);
            }
            double ground_height = Ground(model, data, query);
            double height_target = ground_height + kFootRadius + step[foot];
            double height_difference = foot_pos[foot][2] - height_target;

            if (current_mode_ == kModeScramble) {  // in Scramble, foot higher than target is not penalized
                height_difference = mju_min(0.0, height_difference);
            }

            residual[counter++] = step[foot] ? height_difference : 0.0;
            residual[counter++] = 0.0;
            residual[counter++] = 0.0;
        } else {
            // stance and no active Bezier: no residual
            residual[counter++] = 0.0;
            residual[counter++] = 0.0;
            residual[counter++] = 0.0;
        }
    }

    return counter;
}

Quadruped::ResidualFn::A1Gait Quadruped::ResidualFn::GetGait() const {

    if (current_mode_ == kModeBiped)
        return kGaitTrot;

    return static_cast<A1Gait>(ReinterpretAsInt(current_gait_));
}

double Quadruped::ResidualFn::StepHeight(double time, double footphase,
                                         double duty_ratio) const {

    double angle = fmod(time + mjPI - footphase, 2 * mjPI) - mjPI;
    double value = 0;

    if (duty_ratio < 1) {
        angle *= 0.5 / (1 - duty_ratio);
        value = mju_cos(mju_clip(angle, -mjPI / 2, mjPI / 2));
    }

    return mju_abs(value) < 1e-6 ? 0.0 : value;
}

void Quadruped::ResidualFn::FootStep(double step[kNumFoot], double time,
                                     A1Gait gait) const {

    double amplitude = parameters_[amplitude_param_id_];
    double duty_ratio = parameters_[duty_param_id_];

    for (A1Foot foot : kFootAll) {
        double footphase = 2 * mjPI * kGaitPhase[gait][foot];
        step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
    }
}

double Quadruped::ResidualFn::GetPhase(double time) const {
    return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

}  // namespace mjpc
