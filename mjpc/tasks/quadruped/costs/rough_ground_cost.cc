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
void ComputeBezierTrajectory(const Terrain* terrain, const mjData* data,
                             const double foot_pos[3], const double target_xy[2],
                             double swing_phase, double step_amplitude,
                             double foot_radius, double out_ref_pos[3]) {

    // Feature Extraction
    Terrain::PatchFeatures current_features{};
    Terrain::PatchFeatures target_features{};
    if (terrain) {
        terrain->GetPatchFeatures(data, foot_pos[0], foot_pos[1], current_features);
        terrain->GetPatchFeatures(data, target_xy[0], target_xy[1], target_features);
    }

    // Control Point P3 (Target)
    double P3[3] = {target_xy[0], target_xy[1],
                    target_features.max_height + foot_radius};

    // Control Point P0 (Start) - Calculated via heuristic heading
    double heading[2] = {P3[0] - foot_pos[0], P3[1] - foot_pos[1]};
    double heading_norm = mju_norm(heading, 2);
    if (heading_norm < 1e-6)
        heading_norm = 1.0;
    heading[0] /= heading_norm;
    heading[1] /= heading_norm;
    double step_dist = 0.2;  // heuristic distance
    double P0[3] = {P3[0] - step_dist * heading[0],
                    P3[1] - step_dist * heading[1],
                    current_features.max_height + foot_radius};

    // Control Points P1 & P2 (Clearance Height)
    double mid_x = 0.5 * (P0[0] + P3[0]);
    double mid_y = 0.5 * (P0[1] + P3[1]);
    Terrain::PatchFeatures mid_features{};
    if (terrain) {
        terrain->GetPatchFeatures(data, mid_x, mid_y, mid_features);
    }
    double ground_ref = mju_max(
        mju_max(current_features.max_height, target_features.max_height),
        mid_features.max_height);
    double z_clear = ground_ref + foot_radius + step_amplitude;
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

    if (!IsSwinging(current_step_height)) {
        residual[counter++] = 0.0;
        residual[counter++] = 0.0;
        residual[counter++] = 0.0;

        return counter;
    }

    // set safe target
    double safe_target_xy[2];
    GetProjectedFoothold(data, query, safe_target_xy);

    // get normalized swing phase
    double global_phase = GetPhase(data->time);
    double gait_offset = kGaitPhase[GetGait()][foot];
    double duty_ratio = parameters_[duty_param_id_];
    double swing_phase =
        CalculateNormalizedPhase(global_phase, gait_offset, duty_ratio);

    // generate trajectory
    double reference_pos[3];
    ComputeBezierTrajectory(terrain_, data, foot_pos, safe_target_xy, swing_phase,
                            step_amplitude, kFootRadius, reference_pos);

    // compute residual
    residual[counter++] = foot_pos[0] - reference_pos[0];
    residual[counter++] = foot_pos[1] - reference_pos[1];
    residual[counter++] = foot_pos[2] - reference_pos[2];

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

