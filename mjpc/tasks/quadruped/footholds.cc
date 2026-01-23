#include "mjpc/tasks/quadruped/footholds.h"

#include <cmath>
#include <limits>

#include <mujoco/mujoco.h>

#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/utilities.h"

namespace mjpc {
  

// swing phase per foot in [0,1]
double FootholdPlanner::SwingPhase(double phase, double footphase,
                                   double duty_ratio) {
    if (duty_ratio >= 1.0)
        return 0.0;
    double angle = fmod(phase + mjPI - footphase, 2 * mjPI) - mjPI;
    angle *= 0.5 / (1.0 - duty_ratio);
    angle = mju_clip(angle, -mjPI / 2, mjPI / 2);
    return (angle + mjPI / 2) / mjPI;  // [0,1]
}
 

// is this foot currently in swing?
bool FootholdPlanner::IsSwinging(double phase, double footphase,
                                 double duty_ratio) {
    if (duty_ratio >= 1.0) 
        return false;
    double phi_full = std::fmod(phase - footphase + 2 * mjPI, 2 * mjPI) / (2 * mjPI);  // [0,1)
    const double half_swing = 0.5 * (1.0 - duty_ratio);
    const bool in_stance = (phi_full >= half_swing && phi_full <= 1.0 - half_swing);
    return !in_stance;
}


void FootholdPlanner::ComputeFootholds(const mjModel* model, mjData* data,
                                       const Quadruped::ResidualFn& residual,
                                       double duty_ratio) {
                                        
    // Optional output: publish per-foot world-frame targets for deployment/visualization.
    // IMPORTANT: do not publish zeros for "unused" feet; always publish a meaningful setpoint.
    double* targets = nullptr;
    int targets_dim = 0;
    const int sid = mj_name2id(model, mjOBJ_SENSOR, "foothold_targets");
    if (sid >= 0) {
        targets = data->sensordata + model->sensor_adr[sid];
        targets_dim = model->sensor_dim[sid];
        mju_zero(targets, targets_dim);
    }
    const bool write_targets =
        (targets != nullptr) &&
        (targets_dim >= 3 * Quadruped::ResidualFn::kNumFoot);
    const bool write_userdata =
        (model->nuserdata >= 3 * Quadruped::ResidualFn::kNumFoot);

    // get torso x-direction
    int torso_bid = mj_name2id(model, mjOBJ_XBODY, "trunk");
    const double* torso_mat = data->xmat + 9 * torso_bid;
    double torso_x[3] = {torso_mat[0], torso_mat[3], 0.0};
    mju_normalize3(torso_x);

    // 
    Quadruped::ResidualFn::A1Gait gait = residual.GetGait();
    double step[Quadruped::ResidualFn::kNumFoot];
    double phase = residual.GetPhase(data->time);
    residual.FootStep(step, phase, gait);

    // per-foot unsafe flag
    bool unsafe_ahead[Quadruped::ResidualFn::kNumFoot] = {false, false, false, false};
    // global unsafe is true if at least one foot is unsafe
    bool global_unsafe = false;

  
    for (auto foot : Quadruped::ResidualFn::kFootAll) {
        int gid = residual.foot_geom_id_[foot];
        const double* p0 = data->geom_xpos + 3 * gid;

        // nominal query 15cm in front of the foot
        double query[3] = {p0[0], p0[1], p0[2]};
        mju_addToScl3(query, torso_x, 0.15);
        bool safe = residual.terrain_->IsSafe(data, query[0], query[1]);

        // sample 5 hfield points in front of the foot (XY only for safety check)
        if (safe) {
            const double forward_offsets[5] = {0.04, 0.08, 0.12};
            for (double off : forward_offsets) {
                double sample[3] = {p0[0] + off * torso_x[0],
                                    p0[1] + off * torso_x[1],
                                    p0[2]};
                if (!residual.terrain_->IsSafe(data, sample[0], sample[1])) {
                    safe = false;
                    break;
                }
            }
        }

        unsafe_ahead[foot] = !safe;
        global_unsafe = global_unsafe || !safe;
    }
  

    // actual foothold target computation per foot via Bezier latching
    for (auto foot : Quadruped::ResidualFn::kFootAll) {

        double swing_height = step[foot];
        double footphase = 2 * mjPI * Quadruped::ResidualFn::kGaitPhase[gait][foot];
        bool now_swing = FootholdPlanner::IsSwinging(phase, footphase, duty_ratio);
        int gid = residual.foot_geom_id_[foot];
        const double* foot_pos = data->geom_xpos + 3 * gid;

        // Default published target: hold current foot position (safe for stance and for
        // external consumers that treat the vector as unconditional setpoints).
        if (write_targets) mju_copy3(targets + 3 * foot, foot_pos);
        if (write_userdata) mju_copy3(data->userdata + 3 * foot, foot_pos);

        // If the world is globally safe, immediately drop any latched Beziers and
        // revert to the original "height-only in swing" behavior.
        if (!global_unsafe && !now_swing) {
            bezier_active_[foot] = false;
        }

        // stance: release the previous swing's curve, and latch a new one only in stance.
        if (!now_swing) {
            if (in_swing_[foot]) {
                // Touchdown: clear the swing curve so we can re-latch in stance.
                in_swing_[foot] = false;
                bezier_active_[foot] = false;
            } else {
                in_swing_[foot] = false;
            }

            // If unsafe, design and latch the Bezier during stance so it is ready
            // by the next swing (gait clock).
            if (global_unsafe && !bezier_active_[foot]) {
                double query[3] = {foot_pos[0], foot_pos[1], foot_pos[2]};
                mju_addToScl3(query, torso_x, 0.15);  // nominal forward landing 15cm forward
                const bool unsafe = unsafe_ahead[foot];

                // Global unsafe is true:
                // - if this foot is unsafe: project landing to nearest safe point around nominal query.
                // - if this foot is safe: use nominal query as the landing (trivial Bezier).
                if (unsafe) {
                    const double radii[] = {0.03, 0.05, 0.07, 0.09, 0.12};
                    const int kNumCandidates = 8;
                    double best_dist2 = std::numeric_limits<double>::infinity();
                    double best_xy[2] = {query[0], query[1]};
                    for (double rad : radii) {
                        for (int i = 0; i < kNumCandidates; ++i) {
                            double ang = 2.0 * mjPI * (static_cast<double>(i) / kNumCandidates);
                            double cx = query[0] + rad * mju_cos(ang);
                            double cy = query[1] + rad * mju_sin(ang);
                            if (residual.terrain_->IsSafe(data, cx, cy)) {
                                double dx = cx - query[0];
                                double dy = cy - query[1];
                                double d2 = dx * dx + dy * dy;
                                if (d2 < best_dist2) {
                                    best_dist2 = d2;
                                    best_xy[0] = cx;
                                    best_xy[1] = cy;
                                }
                            }
                        }
                    }

                    query[0] = best_xy[0];
                    query[1] = best_xy[1];
                    double gz = 0.0;
                    residual.terrain_->GetHeightFromWorld(data, query[0], query[1], gz);
                    query[2] = gz + Quadruped::ResidualFn::kFootRadius;
                    mju_copy3(ctrl_pts_[foot][3], query);
                } else {
                    // Trivial landing: just drop the nominal query to the ground.
                    residual.terrain_->GetHeightFromWorld(data, query[0], query[1], query[2]);
                    query[2] += Quadruped::ResidualFn::kFootRadius;
                    mju_copy3(ctrl_pts_[foot][3], query);
                }

                // set control points based on current foot pos and chosen landing
                mju_copy3(ctrl_pts_[foot][0], foot_pos);

                // clearance height: max ground height along the path (4 samples) plus either
                // swing amplitude or 2cm, whichever is larger.
                double max_ground = query[2];
                residual.terrain_->GetHeightFromWorld(data, query[0], query[1], max_ground);
                for (double t : {0.00, 0.33, 0.66, 1.00}) {
                    double sample[3];
                    sample[0] = (1.0 - t) * foot_pos[0] + t * query[0];
                    sample[1] = (1.0 - t) * foot_pos[1] + t * query[1];
                    sample[2] = foot_pos[2];
                    double gz = sample[2];
                    residual.terrain_->GetHeightFromWorld(data, sample[0], sample[1], gz);
                    max_ground = mju_max(max_ground, gz);
                }
                max_ground += Quadruped::ResidualFn::kFootRadius;
                const double lift = mju_max(mju_abs(swing_height), 0.02);
                const double z_clear_final = max_ground + lift;

                ctrl_pts_[foot][1][0] = ctrl_pts_[foot][0][0];
                ctrl_pts_[foot][1][1] = ctrl_pts_[foot][0][1];
                ctrl_pts_[foot][1][2] = z_clear_final;
                ctrl_pts_[foot][2][0] = ctrl_pts_[foot][3][0];
                ctrl_pts_[foot][2][1] = ctrl_pts_[foot][3][1];
                ctrl_pts_[foot][2][2] = z_clear_final;

                bezier_active_[foot] = true;
            }

            continue;
        }

        // swing entry: mark swing, but do not latch a new curve during swing.
        if (!in_swing_[foot]) {
            in_swing_[foot] = true;
        }

        double swing_phase = FootholdPlanner::SwingPhase(phase, footphase, duty_ratio);

        // If a Bezier is already latched, just evaluate it.
        if (bezier_active_[foot]) {
            double out[3];
            EvalBezier(foot, swing_phase, out);
            if (write_targets) mju_copy3(targets + 3 * foot, out);
            if (write_userdata) mju_copy3(data->userdata + 3 * foot, out);
            continue;
        }

        // No latched Bezier yet: publish the same height-only swing target used by the
        // MJPC residual (vertical clearance), keeping XY at the current foot position.
        if (write_targets || write_userdata) {
            double tgt[3] = {foot_pos[0], foot_pos[1], foot_pos[2]};
            double ground_z = foot_pos[2];
            if (residual.terrain_) {
                residual.terrain_->GetHeightFromWorld(data, tgt[0], tgt[1], ground_z);
            }
            tgt[2] = ground_z + Quadruped::ResidualFn::kFootRadius + swing_height;
            if (write_targets) mju_copy3(targets + 3 * foot, tgt);
            if (write_userdata) mju_copy3(data->userdata + 3 * foot, tgt);
        }
    }
}

void FootholdPlanner::EvalBezier(Quadruped::ResidualFn::A1Foot foot,
                                 double t, double out[3]) const {
                                            
    const double (*ctrl)[3] = ctrl_pts_[foot];

    const double one = 1.0 - t;
    const double one2 = one * one;
    const double t2 = t * t;

    // Bernstein basis for cubic Bezier:
    const double b0 = one2 * one;
    const double b1 = 3.0 * one2 * t;
    const double b2 = 3.0 * one * t2;
    const double b3 = t * t2;

    // out = Σ bi * Pi
    mju_scl3(out, ctrl[0], b0);
    mju_addToScl3(out, ctrl[1], b1);
    mju_addToScl3(out, ctrl[2], b2);
    mju_addToScl3(out, ctrl[3], b3);
}

}  // namespace mjpc

