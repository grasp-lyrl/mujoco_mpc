#include "mjpc/tasks/quadruped/footholds.h"

#include <cmath>

#include <mujoco/mujoco.h>

#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace {
// latched control points and swing state
double g_ctrl_pts[Quadruped::ResidualFn::kNumFoot][4][3] = {{{0}}};
bool g_in_swing[Quadruped::ResidualFn::kNumFoot] = {false, false, false, false};
bool g_bezier_active[Quadruped::ResidualFn::kNumFoot] = {false, false, false, false};
}  // namespace

namespace {
// Ground helper with graceful fallback if ray misses any group-0 geom.
double SafeGround(const mjModel* model, const mjData* data, const double pos[3]) {
  mjtNum down[3] = {0, 0, -1};
  const mjtNum height_offset = .5;
  const mjtByte flg_static = 1;
  const int bodyexclude = -1;
  const mjtByte default_geomgroup[6] = {1, 0, 0, 0, 0, 0};
  mjtNum query[3] = {pos[0], pos[1], pos[2] + height_offset};
  int geomid;
  mjtNum dist = mj_ray(model, data, query, down, default_geomgroup, flg_static,
                       bodyexclude, &geomid);
  if (dist < 0) return pos[2];  // fallback: keep current height
  return pos[2] + height_offset - dist;
}
}  // namespace

// swing phase aligned with StepHeight cylinders (same as visuals).
double SwingPhase(double phase, double footphase, double duty_ratio) {
  if (duty_ratio >= 1.0) return 0.0;
  double angle = fmod(phase + mjPI - footphase, 2 * mjPI) - mjPI;
  angle *= 0.5 / (1.0 - duty_ratio);
  angle = mju_clip(angle, -mjPI / 2, mjPI / 2);
  return (angle + mjPI / 2) / mjPI;  // [0,1]
}

// swing state aligned with the gait cylinder clock (no flicker at step=0 crossings)
bool PhaseInSwing(double phase, double footphase, double duty_ratio) {
  if (duty_ratio >= 1.0) return false;  // no swing when duty>=1
  double phi_full =
      std::fmod(phase - footphase + 2 * mjPI, 2 * mjPI) / (2 * mjPI);  // [0,1)
  const double half_swing = 0.5 * (1.0 - duty_ratio);
  const bool in_stance = (phi_full >= half_swing && phi_full <= 1.0 - half_swing);
  return !in_stance;
}

void ComputeFootholdTarget(const mjModel* model, const mjData* data,
                           const Quadruped::ResidualFn& residual,
                           Quadruped::ResidualFn::A1Foot foot,
                           Quadruped::ResidualFn::A1Gait gait,
                           double phase, double duty_ratio, double step_height,
                           const double torso_x[3], double out_target[3]) {
  (void)model;
  (void)data;
  (void)residual;
  (void)foot;
  (void)gait;
  (void)phase;
  (void)duty_ratio;
  (void)step_height;
  (void)torso_x;
  mju_zero3(out_target);
}

void ComputeFootholds(const mjModel* model, mjData* data,
                      const Quadruped::ResidualFn& residual,
                      double duty_ratio) {
  int sid = mj_name2id(model, mjOBJ_SENSOR, "foothold_targets");
  if (sid < 0) return;
  int adr = model->sensor_adr[sid];
  int dim = model->sensor_dim[sid];
  double* targets = data->sensordata + adr;
  mju_zero(targets, dim);

  int torso_bid = mj_name2id(model, mjOBJ_XBODY, "trunk");
  const double* torso_mat = data->xmat + 9 * torso_bid;
  // Use torso x-axis column as forward (ignore z component).
  double torso_x[3] = {torso_mat[0], torso_mat[3], 0.0};
  mju_normalize3(torso_x);

  Quadruped::ResidualFn::A1Gait gait = residual.GetGait();
  double step[Quadruped::ResidualFn::kNumFoot];
  double phase = residual.GetPhase(data->time);
  residual.FootStep(step, phase, gait);

  // Precompute per-foot "unsafe ahead" and a global flag.
  // When global_unsafe is true, every foot will track a (possibly trivial) Bezier
  // until the world becomes globally safe again.
  bool unsafe_ahead[Quadruped::ResidualFn::kNumFoot] = {false, false, false, false};
  bool global_unsafe = false;
  if (residual.terrain_) {
    for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
      int gid = residual.foot_geom_id_[foot];
      const double* p0 = data->geom_xpos + 3 * gid;

      // nominal query 15cm in front of the foot (same as visuals)
      double query[3] = {p0[0], p0[1], p0[2]};
      mju_addToScl3(query, torso_x, 0.15);
      bool unsafe = !residual.terrain_->IsSafe(data, query[0], query[1]);

      // sample 5 hfield points in front of the foot (XY only for safety check)
      if (!unsafe) {
        const double forward_offsets[5] = {0.03, 0.06, 0.09, 0.12, 0.15};
        for (double off : forward_offsets) {
          double sample[3] = {p0[0] + off * torso_x[0],
                              p0[1] + off * torso_x[1],
                              p0[2]};
          if (!residual.terrain_->IsSafe(data, sample[0], sample[1])) {
            unsafe = true;
            break;
          }
        }
      }

      unsafe_ahead[foot] = unsafe;
      global_unsafe = global_unsafe || unsafe;
    }
  }

  for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
    double swing_height = step[foot];
    double footphase = 2 * mjPI * Quadruped::ResidualFn::kGaitPhase[gait][foot];
    bool now_swing = PhaseInSwing(phase, footphase, duty_ratio);
    int gid = residual.foot_geom_id_[foot];
    const double* p0 = data->geom_xpos + 3 * gid;

    // If the world is globally safe, immediately drop any latched Beziers and
    // revert to the original "height-only in swing" behavior.
    if (!global_unsafe) {
      g_bezier_active[foot] = false;
    }

    // stance: keep holding the last landing target if a Bezier was latched,
    // but do not clear the latch here (it is released at the next swing start).
    if (!now_swing) {
      g_in_swing[foot] = false;
      if (g_bezier_active[foot]) {
        // On touchdown, latch the true contact point as the stance target.
        mju_copy3(g_ctrl_pts[foot][3], p0);
        mju_copy3(targets + 3 * foot, g_ctrl_pts[foot][3]);  // hold landing
        if (data->userdata) mju_copy3(data->userdata + 3 * foot, targets + 3 * foot);
      } else {
        mju_zero3(targets + 3 * foot);
        if (data->userdata) mju_zero3(data->userdata + 3 * foot);
      }
      continue;
    }

    // swing entry: release any previously-held landing so a new curve can be designed
    if (!g_in_swing[foot]) {
      g_in_swing[foot] = true;
      g_bezier_active[foot] = false;
    }

    double swing_phase = SwingPhase(phase, footphase, duty_ratio);

    // If a Bezier is already latched, just evaluate it
    if (g_bezier_active[foot]) {
      double one = 1.0 - swing_phase;
      double one2 = one * one;
      double phi2 = swing_phase * swing_phase;
      double b0 = one2 * one;
      double b1 = 3 * one2 * swing_phase;
      double b2 = 3 * one * phi2;
      double b3 = swing_phase * phi2;
      for (int k = 0; k < 3; ++k) {
        targets[3 * foot + k] =
            b0 * g_ctrl_pts[foot][0][k] + b1 * g_ctrl_pts[foot][1][k] +
            b2 * g_ctrl_pts[foot][2][k] + b3 * g_ctrl_pts[foot][3][k];
      }
      if (data->userdata) mju_copy3(data->userdata + 3 * foot, targets + 3 * foot);
      continue;
    }

    // No latched Bezier yet: either latch due to per-foot unsafe, or (if any foot
    // is unsafe) latch a trivial Bezier so all feet track a consistent trajectory.
    double query[3] = {p0[0], p0[1], p0[2]};
    mju_addToScl3(query, torso_x, 0.15);  // nominal forward landing (same as visuals)
    const bool unsafe = unsafe_ahead[foot];

    if (!global_unsafe) {
      // globally safe: no 3D targets; let gait cost be height-only in swing
      mju_zero3(targets + 3 * foot);
      if (data->userdata) mju_zero3(data->userdata + 3 * foot);
      continue;
    }

    // Global unsafe is true:
    // - if this foot is unsafe: project landing to nearest safe point around nominal query.
    // - if this foot is safe: use nominal query as the landing (trivial Bezier).
    if (unsafe && residual.terrain_) {
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
      mju_copy3(g_ctrl_pts[foot][3], query);
    } else {
      // Trivial landing: just drop the nominal query to the ground.
      query[2] = SafeGround(model, data, query) + Quadruped::ResidualFn::kFootRadius;
      mju_copy3(g_ctrl_pts[foot][3], query);
    }

    // set control points based on current foot pos and chosen landing
    mju_copy3(g_ctrl_pts[foot][0], p0);

    // clearance height: max ground height along the path (4 samples) plus either
    // swing amplitude or 2cm, whichever is larger.
    double max_ground = SafeGround(model, data, query);
    for (double t : {0.00, 0.33, 0.66, 1.00}) {
      double sample[3];
      sample[0] = (1.0 - t) * p0[0] + t * query[0];
      sample[1] = (1.0 - t) * p0[1] + t * query[1];
      sample[2] = p0[2];
      max_ground = mju_max(max_ground, SafeGround(model, data, sample));
    }
    max_ground += Quadruped::ResidualFn::kFootRadius;
    const double lift = mju_max(mju_abs(swing_height), 0.02);
    const double z_clear_final = max_ground + lift;

    g_ctrl_pts[foot][1][0] = g_ctrl_pts[foot][0][0];
    g_ctrl_pts[foot][1][1] = g_ctrl_pts[foot][0][1];
    g_ctrl_pts[foot][1][2] = z_clear_final;
    g_ctrl_pts[foot][2][0] = g_ctrl_pts[foot][3][0];
    g_ctrl_pts[foot][2][1] = g_ctrl_pts[foot][3][1];
    g_ctrl_pts[foot][2][2] = z_clear_final;

    double one = 1.0 - swing_phase;
    double one2 = one * one;
    double phi2 = swing_phase * swing_phase;
    double b0 = one2 * one;
    double b1 = 3 * one2 * swing_phase;
    double b2 = 3 * one * phi2;
    double b3 = swing_phase * phi2;
    for (int k = 0; k < 3; ++k) {
      targets[3 * foot + k] =
          b0 * g_ctrl_pts[foot][0][k] + b1 * g_ctrl_pts[foot][1][k] +
          b2 * g_ctrl_pts[foot][2][k] + b3 * g_ctrl_pts[foot][3][k];
    }
    if (data->userdata) {
      mju_copy3(data->userdata + 3 * foot, targets + 3 * foot);
    }
    g_bezier_active[foot] = true;
  }
}

bool GetLatchedControlPoints(Quadruped::ResidualFn::A1Foot foot,
                             double ctrl_pts[4][3]) {
  for (int i = 0; i < 4; ++i) mju_copy3(ctrl_pts[i], g_ctrl_pts[foot][i]);
  return g_bezier_active[foot];
}

bool IsFootSwinging(Quadruped::ResidualFn::A1Foot foot) {
  return g_in_swing[foot];
}

bool IsBezierActive(Quadruped::ResidualFn::A1Foot foot) {
  return g_bezier_active[foot];
}

void EvalLatchedBezier(Quadruped::ResidualFn::A1Foot foot, double t,
                       double out[3]) {
  double ctrl[4][3];
  GetLatchedControlPoints(foot, ctrl);
  double o = 1.0 - t;
  double o2 = o * o;
  double t2 = t * t;
  double b0 = o2 * o;
  double b1 = 3 * o2 * t;
  double b2 = 3 * o * t2;
  double b3 = t * t2;
  for (int k = 0; k < 3; ++k) {
    out[k] = b0 * ctrl[0][k] + b1 * ctrl[1][k] + b2 * ctrl[2][k] + b3 * ctrl[3][k];
  }
}

}  // namespace mjpc

