#include "mjpc/tasks/quadruped/quadruped.h"

#include <mujoco/mujoco.h>

#include <cmath>
#include <limits>

namespace mjpc {


inline bool IsSwinging(double current_step_height) {
    return current_step_height > 1e-4;
}

// normalized swing phase (0.0 -> 1.0)
double CalculateNormalizedPhase(double global_phase, 
                                double gait_phase_offset,
                                double duty_ratio) {

    double foot_phase = 2 * mjPI * gait_phase_offset;

    double phi_full =
        std::fmod(global_phase - foot_phase + 2 * mjPI, 2 * mjPI) / (2 * mjPI);
    double swing_phase = (phi_full - duty_ratio) / (1.0 - duty_ratio);
    return mju_clip(swing_phase, 0.0, 1.0);
}

// Compute Cubic Bezier Curve (Physics Heuristic)
// P0: 5cm in front of the hip (torso +X), plus a torso +Y lateral offset
//     (left/right), at
//     terrain height + foot radius.
// P3: safe target XY, at terrain height + foot radius.
void ComputeBezierTrajectory(const Terrain* terrain, const mjData* data,
                             const double hip_pos[3], const double torso_x[3],
                             const double torso_y[3],
                             double lateral_sign, const double target_xy[2],
                             double swing_phase, double step_amplitude,
                             double foot_radius, double out_ref_pos[3]) {

    // Feature Extraction
    // P0 XY: forward offset from torso +X axis + lateral offset from torso +Y axis
    constexpr double kHipForwardOffset = 0.05;  // 5cm
    constexpr double kHipLateralOffset = 0.10;  // 10cm
    double P0_xy[2] = {hip_pos[0] + kHipForwardOffset * torso_x[0] +
                           lateral_sign * kHipLateralOffset * torso_y[0],
                       hip_pos[1] + kHipForwardOffset * torso_x[1] +
                           lateral_sign * kHipLateralOffset * torso_y[1]};

    Terrain::PatchFeatures start_features{};
    Terrain::PatchFeatures target_features{};
    if (terrain) {
        terrain->GetPatchFeatures(data, P0_xy[0], P0_xy[1], start_features);
        terrain->GetPatchFeatures(data, target_xy[0], target_xy[1], target_features);
    }

    // Control Point P3 (Target)
    double P3[3] = {target_xy[0], target_xy[1],
                    target_features.max_height + foot_radius};

    // Control Point P0 (Start) - hip + 5cm forward (torso +X) + lateral offset.
    double P0[3] = {P0_xy[0], P0_xy[1], start_features.max_height + foot_radius};

    // Control Points P1 & P2 (Clearance Height)
    double mid_x = 0.5 * (P0[0] + P3[0]);
    double mid_y = 0.5 * (P0[1] + P3[1]);
    Terrain::PatchFeatures mid_features{};
    if (terrain) {
        terrain->GetPatchFeatures(data, mid_x, mid_y, mid_features);
    }
    double ground_ref = mju_max(
        mju_max(start_features.max_height, target_features.max_height),
        mid_features.max_height);
    // Adaptive clearance boost: increase apex height over curbs/steep patches.
    // - obstacle_height: vertical discontinuity along the path
    // - slope_factor: low normal-z indicates steepness
    constexpr double kObstacleClearanceGain = 0.5;   // m per m of obstacle
    constexpr double kSlopeClearanceGain = 0.05;     // m per unit of (1-normal_z)
    constexpr double kMaxExtraClearance = 0.10;      // m (safety clamp)
    double low_ref =
        mju_min(start_features.max_height, target_features.max_height);
    double obstacle_height = mju_max(0.0, ground_ref - low_ref);
    double min_normal_z = mju_min(
        mju_min(start_features.normal[2], target_features.normal[2]),
        mid_features.normal[2]);
    double slope_factor = mju_max(0.0, 1.0 - min_normal_z);
    double extra_clear =
        kObstacleClearanceGain * obstacle_height + kSlopeClearanceGain * slope_factor;
    extra_clear = mju_min(extra_clear, kMaxExtraClearance);

    double z_clear = ground_ref + foot_radius + step_amplitude + extra_clear;
    double P1[3] = {P0[0], P0[1], z_clear};
    double P2[3] = {P3[0], P3[1], z_clear};

    // Bernstein Polynomial Interpolation
    double one = 1.0 - swing_phase;
    double one2 = one * one;
    double phi2 = swing_phase * swing_phase;

    double b0 = one2 * one;             // (1-t)^3
    double b1 = 3 * one2 * swing_phase; // 3(1-t)^2 * t
    double b2 = 3 * one * phi2;         // 3(1-t) * t^2
    double b3 = swing_phase * phi2;     // t^3
    for (int i = 0; i < 3; ++i) {
        out_ref_pos[i] = b0 * P0[i] + b1 * P1[i] + b2 * P2[i] + b3 * P3[i];
    }
}

// main
int Quadruped::ResidualFn::CostRoughGround(const mjModel* model,
                                           const mjData* data, A1Foot foot,
                                           const double foot_pos[3],
                                           const double query[3],
                                           double step_amplitude,
                                           double current_step_height,
                                           double* residual,
                                           int counter) const {
    bool is_swinging = IsSwinging(current_step_height);
    // set safe target
    double safe_target_xy[2];
    GetProjectedFoothold(data, query, safe_target_xy);

    // Get swing phase parameter aligned with StepHeight() (mustard cylinders).
    // This ensures the Bezier apex occurs when the gait visualization is maximal.
    double swing_phase = 0.0;
    double duty_ratio = parameters_[duty_param_id_];
    if (duty_ratio < 1.0) {
        double phase = GetPhase(data->time);
        double footphase = 2 * mjPI * kGaitPhase[GetGait()][foot];
        double angle = fmod(phase + mjPI - footphase, 2 * mjPI) - mjPI;
        angle *= 0.5 / (1.0 - duty_ratio);
        angle = mju_clip(angle, -mjPI / 2, mjPI / 2);
        swing_phase = (angle + mjPI / 2) / mjPI;  // [-pi/2,pi/2] -> [0,1]
    }

    // generate trajectory anchored at hip + 5cm forward (torso +X) + torso-lateral offset
    const double* hip_pos = data->xipos + 3 * shoulder_body_id_[foot];
    const double* torso_mat = data->xmat + 9 * torso_body_id_;
    double torso_x[3] = {torso_mat[0], torso_mat[1], torso_mat[2]};
    double torso_y[3] = {torso_mat[3], torso_mat[4], torso_mat[5]};
    double lateral_sign = (foot == kFootFR || foot == kFootHR) ? -1.0 : 1.0;
    double reference_pos[3];
    ComputeBezierTrajectory(terrain_, data, hip_pos, torso_x, torso_y, lateral_sign,
                            safe_target_xy, swing_phase, step_amplitude,
                            kFootRadius, reference_pos);

    // compute residual
    // Emphasize XY tracking relative to Z: double the XY residual magnitude.
    // Note: the Gait term uses SmoothAbsLoss (norm=6), so this increases XY
    // penalty approximately 2x for larger errors (and >2x in the very small-error
    // quadratic regime).
    residual[counter++] = foot_pos[0] - reference_pos[0];
    residual[counter++] = foot_pos[1] - reference_pos[1];
    double z_err = foot_pos[2] - reference_pos[2];
    if (!is_swinging) {
        // In stance, pull down only toward a floor-clamped target to avoid
        // driving the foot below terrain.
        double ground_z = 0.0;
        terrain_->GetHeightFromWorld(data, safe_target_xy[0], safe_target_xy[1],
                                     ground_z);
        double floor_target = ground_z + kFootRadius;
        double desired_z = mju_max(reference_pos[2], floor_target);
        z_err = mju_max(foot_pos[2] - desired_z, 0.0);
    }
    residual[counter++] = z_err;

    return counter;
}

void Quadruped::ResidualFn::GetProjectedFoothold(const mjData* data,
                                                 const double query[3],
                                                 double safe_xy[2]) const {

    if (terrain_->IsSafe(data, query[0], query[1])) {
        safe_xy[0] = query[0];
        safe_xy[1] = query[1];
        return;
    }

    // concentric multi-ring search
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

}  // namespace mjpc

