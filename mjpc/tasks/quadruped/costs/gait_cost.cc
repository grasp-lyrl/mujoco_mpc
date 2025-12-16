#include "mjpc/tasks/quadruped/quadruped.h"
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
    double target_amp = parameters_[amplitude_param_id_];

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

        double query[3] = {foot_pos[foot][0], 
                           foot_pos[foot][1],
                           foot_pos[foot][2]};

        if (current_mode_ == kModeScramble) {
            double torso_to_goal[3];
            double* goal = goal_pos;
            mju_sub3(torso_to_goal, goal, torso_pos);
            mju_normalize3(torso_to_goal);
            torso_to_goal[2] = 0;
            mju_normalize3(torso_to_goal);
            mju_addToScl3(query, torso_to_goal, 0.15);
        }

        bool use_rough = ReinterpretAsInt(parameters_[terrain_type_param_id_]);
        if (use_rough) {
            counter = CostRoughGround(model, data, foot, foot_pos[foot], query,
                                      target_amp, step[foot], residual, counter);
        } else {
            // flat terrain: mimic original gait cost behavior (swing height from step[foot])
            counter = CostFlatGround(model, data, foot, foot_pos[foot], query,
                                     step[foot], residual, counter);
        }
    }

    return counter;
}

int Quadruped::ResidualFn::CostFlatGround(const mjModel* model,
                                          const mjData* data, A1Foot foot,
                                          const double foot_pos[3],
                                          const double query[3],
                                          double step_amplitude,
                                          double* residual,
                                          int counter) const {
    double ground_height = Ground(model, data, query);
    double height_target = ground_height + kFootRadius + step_amplitude;
    double height_difference = foot_pos[2] - height_target;

    if (current_mode_ == kModeScramble) {  // in Scramble, foot higher than target is not penalized
        height_difference = mju_min(0.0, height_difference);
    }

    residual[counter++] = step_amplitude ? height_difference : 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
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
