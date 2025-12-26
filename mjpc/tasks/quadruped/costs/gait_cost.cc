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
    // Keep the query anchor consistent with the Bezier P0 shift used in rough terrain.
    constexpr double kHipForwardOffset = 0.05;  // 5cm along torso +X
    const double* torso_mat = data->xmat + 9 * torso_body_id_;
    double torso_x[3] = {torso_mat[0], torso_mat[1], torso_mat[2]};

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

        // Anchor target at the hip and push it 10cm outward along the hip +Y
        // axis so footholds land laterally away from the body.
        double* hip_pos = data->xipos + 3 * shoulder_body_id_[foot];
        const double* hip_mat = data->xmat + 9 * shoulder_body_id_[foot];
        double lateral = (foot == kFootFR || foot == kFootHR) ? -0.10 : 0.10;
        double hip_offset_world[3] = {
            hip_mat[3] * lateral,  // +Y column
            hip_mat[4] * lateral,
            hip_mat[5] * lateral};
        double query[3] = {hip_pos[0] + hip_offset_world[0] + kHipForwardOffset * torso_x[0],
                           hip_pos[1] + hip_offset_world[1] + kHipForwardOffset * torso_x[1],
                           hip_pos[2] + hip_offset_world[2]};

        if (current_mode_ == kModeScramble) {
            // Behavior (exactly as requested):
            // - Stance (cylinder==0, touchdown -> liftoff): first 50% under foot, next 50% teleport to +15cm.
            // - Swing (cylinder>0, liftoff -> touchdown): first 60% hold +15cm, last 40% retract linearly to 0.
            double duty_ratio = parameters_[duty_param_id_];
            double scale = 0.0;
            if (duty_ratio < 1.0) {
                double global_phase = GetPhase(data->time);
                double foot_phase = 2 * mjPI * kGaitPhase[gait][foot];
                double phi_full =
                    std::fmod(global_phase - foot_phase + 2 * mjPI, 2 * mjPI) /
                    (2 * mjPI);  // [0,1)

                // StepHeight() is nonzero only when |wrapped_phase| < (1-duty_ratio)/2 (around phi_full=0).
                // That means swing occurs for phi_full in [0, half_swing] U [1-half_swing, 1).
                const double half_swing = 0.5 * (1.0 - duty_ratio);
                const bool in_stance = (phi_full >= half_swing && phi_full <= 1.0 - half_swing);

                if (in_stance) {
                    // stance is contiguous: touchdown at phi_full=half_swing, liftoff at phi_full=1-half_swing
                    double stance_progress = (phi_full - half_swing) / duty_ratio;  // [0,1]
                    if (stance_progress < 0.50) {
                        scale = 0.0;
                    } else {
                        // linearly ramp from under-foot to 15cm over the second half of stance
                        double t = (stance_progress - 0.50) / 0.50;  // [0,1]
                        t = mju_clip(t, 0.0, 1.0);
                        scale = t;
                    }
                } else {
                    // swing phase aligned to cylinder: 0 at liftoff, 0.5 at peak, 1 at touchdown
                    double angle = fmod(global_phase + mjPI - foot_phase, 2 * mjPI) - mjPI;
                    angle *= 0.5 / (1.0 - duty_ratio);
                    angle = mju_clip(angle, -mjPI / 2, mjPI / 2);
                    double swing_phase = (angle + mjPI / 2) / mjPI;  // [0,1]

                    if (swing_phase < 0.60) {
                        scale = 1.0;
                    } else {
                        double t = (swing_phase - 0.60) / 0.40;  // [0,1]
                        t = mju_clip(t, 0.0, 1.0);
                        scale = 1.0 - t;
                    }
                }
            }

            double torso_to_goal[3];
            double* goal = goal_pos;
            mju_sub3(torso_to_goal, goal, torso_pos);
            mju_normalize3(torso_to_goal);
            mju_sub3(torso_to_goal, goal, hip_pos);
            torso_to_goal[2] = 0;
            mju_normalize3(torso_to_goal);
            mju_addToScl3(query, torso_to_goal, 0.15 * scale);
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
