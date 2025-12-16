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

        double query[3] = {foot_pos[foot][0], foot_pos[foot][1],
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

int Quadruped::ResidualFn::CostRoughGround(const mjModel* model,
                                           const mjData* data, A1Foot foot,
                                           const double foot_pos[3],
                                           const double query[3],
                                           double step_amplitude,
                                           double current_step_height,
                                           double* residual,
                                           int counter) const {
    if (current_step_height == 0.0) {
        residual[counter++] = 0.0;
        residual[counter++] = 0.0;
        residual[counter++] = 0.0;
        return counter;
    }

    double safe_xy[2];
    GetProjectedFoothold(data, query, safe_xy);

    Terrain::PatchFeatures target_features{};
    terrain_->GetPatchFeatures(data, safe_xy[0], safe_xy[1], target_features);

    Terrain::PatchFeatures current_features{};
    terrain_->GetPatchFeatures(data, foot_pos[0], foot_pos[1], current_features);

    // swing phase remapped to swing portion only
    double phase = GetPhase(data->time);
    double footphase = 2 * mjPI * kGaitPhase[GetGait()][foot];
    double phi_full = std::fmod(phase - footphase + 2 * mjPI, 2 * mjPI) / (2 * mjPI);
    double duty_ratio = parameters_[duty_param_id_];
    double swing_phase = (phi_full - duty_ratio) / (1.0 - duty_ratio);
    swing_phase = mju_clip(swing_phase, 0.0, 1.0);

    // control points
    double P3[3] = {safe_xy[0], safe_xy[1], target_features.max_height + kFootRadius};

    // heuristic start point (fallback to using foot position for XY)
    double step_dist = 0.2;
    double heading[2] = {P3[0] - foot_pos[0], P3[1] - foot_pos[1]};
    double heading_norm = mju_norm(heading, 2);
    if (heading_norm < 1e-6) heading_norm = 1.0;
    heading[0] /= heading_norm;
    heading[1] /= heading_norm;
    double P0[3] = {P3[0] - step_dist * heading[0],
                    P3[1] - step_dist * heading[1],
                    current_features.max_height + kFootRadius};

    // midpoint sampling for conservative clearance
    double mid_x = 0.5 * (P0[0] + P3[0]);
    double mid_y = 0.5 * (P0[1] + P3[1]);
    Terrain::PatchFeatures mid_features{};
    terrain_->GetPatchFeatures(data, mid_x, mid_y, mid_features);
    double midpoint_z = mid_features.max_height;

    double ground_ref = mju_max(mju_max(current_features.max_height, target_features.max_height),
                                midpoint_z);
    double z_clear = ground_ref + kFootRadius + step_amplitude;
    double P1[3] = {P0[0], P0[1], z_clear};
    double P2[3] = {P3[0], P3[1], z_clear};

    double one = 1.0 - swing_phase;
    double one2 = one * one;
    double phi2 = swing_phase * swing_phase;
    double b0 = one2 * one;
    double b1 = 3 * one2 * swing_phase;
    double b2 = 3 * one * phi2;
    double b3 = swing_phase * phi2;

    double pref[3];
    for (int i = 0; i < 3; ++i) {
        pref[i] = b0 * P0[i] + b1 * P1[i] + b2 * P2[i] + b3 * P3[i];
    }

    double swing_gate = current_step_height > 1e-4 ? 1.0 : 0.0;
    residual[counter++] = swing_gate * (foot_pos[0] - pref[0]);
    residual[counter++] = swing_gate * (foot_pos[1] - pref[1]);
    residual[counter++] = swing_gate * (foot_pos[2] - pref[2]);
    return counter;
}

void Quadruped::ResidualFn::GetProjectedFoothold(const mjData* data,
                                                 const double query[3],
                                                 double safe_xy[2]) const {
    if (!terrain_) {
        safe_xy[0] = query[0];
        safe_xy[1] = query[1];
        return;
    }

    if (terrain_->IsSafe(data, query[0], query[1])) {
        safe_xy[0] = query[0];
        safe_xy[1] = query[1];
        return;
    }

    // concentric multi-ring search around the query
    safe_xy[0] = query[0];
    safe_xy[1] = query[1];
    double min_dist_sq = std::numeric_limits<double>::infinity();

    constexpr int kNumCandidates = 8;
    const double radii[] = {0.03, 0.05, 0.07, 0.09, 0.12};
    for (double rad : radii) {
        for (int i = 0; i < kNumCandidates; ++i) {
            double angle = 2.0 * mjPI * (static_cast<double>(i) / kNumCandidates);
            double cx = query[0] + rad * mju_cos(angle);
            double cy = query[1] + rad * mju_sin(angle);

            if (terrain_->IsSafe(data, cx, cy)) {
                double dx = cx - query[0];
                double dy = cy - query[1];
                double dist_sq = dx * dx + dy * dy;
                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    safe_xy[0] = cx;
                    safe_xy[1] = cy;
                }
            }
        }
    }
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
