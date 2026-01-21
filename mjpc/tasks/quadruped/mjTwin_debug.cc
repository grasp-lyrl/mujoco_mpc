#include "mjpc/tasks/quadruped/mjTwin_debug.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>

#include "mjpc/utilities.h"

namespace mjpc {

namespace {

constexpr double kBezierStride = 0.12;  // meters forward from the initial FR foot

void EvalBezier(const double ctrl[4][3], double t, double out[3]) {
  const double one = 1.0 - t;
  const double one2 = one * one;
  const double t2 = t * t;
  const double b0 = one2 * one;
  const double b1 = 3.0 * one2 * t;
  const double b2 = 3.0 * one * t2;
  const double b3 = t * t2;

  out[0] = b0 * ctrl[0][0] + b1 * ctrl[1][0] + b2 * ctrl[2][0] + b3 * ctrl[3][0];
  out[1] = b0 * ctrl[0][1] + b1 * ctrl[1][1] + b2 * ctrl[2][1] + b3 * ctrl[3][1];
  out[2] = b0 * ctrl[0][2] + b1 * ctrl[1][2] + b2 * ctrl[2][2] + b3 * ctrl[3][2];
}

}  // namespace

std::string MjTwinDebug::XmlPath() const {
  return GetModelPath("quadruped/xmls/task_mjTwin_debug.xml");
}

std::string MjTwinDebug::Name() const { return "mjTwin_debug"; }

// Keep foothold visualization (orange curve + purple target) consistent with the
// actual target used in the residual.
void MjTwinDebug::TransitionEnvOnlyLocked(mjModel* model, mjData* data) {
  // Keep MjTwin's environment updates (collision boxes + unsafe overlay).
  MjTwin::TransitionEnvOnlyLocked(model, data);

  // Mirror the base residual state so GetPhase/GetGait match MjTwin.
  debug_residual_.CopyFrom(residual_, this);

  if (!initialized_) return;

  // Keep the goal mocap fixed at initial torso position (torso stay-in-place).
  if (debug_residual_.goal_mocap_id_ >= 0) {
    mju_copy3(data->mocap_pos + 3 * debug_residual_.goal_mocap_id_, initial_torso_pos_);
  }

  // Force-enable a Bezier for FR so MjTwin's visualizer draws the curve.
  const int fr = Quadruped::ResidualFn::kFootFR;
  for (int i = 0; i < 4; ++i) {
    mju_copy3(foothold_planner_.ctrl_pts_[fr][i], fr_bezier_ctrl_[i]);
  }
  for (int f = 0; f < Quadruped::ResidualFn::kNumFoot; ++f) {
    foothold_planner_.bezier_active_[f] = (f == fr);
  }

  // Compute the *exact* target point we want to track and publish it as the
  // foothold_targets user-sensor (this is what the purple sphere visualizes).
  int sid = mj_name2id(model, mjOBJ_SENSOR, "foothold_targets");
  if (sid >= 0) {
    double* targets = data->sensordata + model->sensor_adr[sid];
    mju_zero(targets, model->sensor_dim[sid]);

    const int gait = static_cast<int>(debug_residual_.GetGait());
    if (gait >= 0 && gait < Quadruped::ResidualFn::kNumGait) {
      const double phase = debug_residual_.GetPhase(data->time);
      const double footphase =
          2 * mjPI * Quadruped::ResidualFn::kGaitPhase[gait][fr];
      const double t = 0.5 * (1.0 - std::cos(phase - footphase));
      EvalBezier(fr_bezier_ctrl_, t, targets + 3 * fr);
      // keep userdata in sync too (optional debug/storage)
      mju_copy3(data->userdata + 3 * fr, targets + 3 * fr);
    }
  }
}

void MjTwinDebug::ResetLocked(const mjModel* model) {
  MjTwin::ResetLocked(model);
  initialized_ = false;
  
  // Update both residuals to ensure they have the latest task parameters/weights
  residual_.Update();
  debug_residual_.CopyFrom(residual_, this);
  debug_residual_.Update();
}

void MjTwinDebug::TransitionLocked(mjModel* model, mjData* data) {
  // Synchronize inherited residual before MjTwin uses it
  residual_.Update();
  MjTwin::TransitionLocked(model, data);
  
  // Synchronize debug residual after MjTwin might have modified residual_
  debug_residual_.CopyFrom(residual_, this);
  debug_residual_.Update();

  if (debug_residual_.torso_body_id_ < 0) {
    return;
  }
  for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
    if (debug_residual_.foot_geom_id_[foot] < 0) {
      return;
    }
  }

  if (!initialized_) {
    for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
      const double* foot_pos = data->geom_xpos + 3 * debug_residual_.foot_geom_id_[foot];
      mju_copy3(initial_foot_pos_[foot], foot_pos);
    }

    // Fix the torso target at the current torso position to keep the body in place.
    const double* torso_pos = data->xipos + 3 * debug_residual_.torso_body_id_;
    mju_copy3(initial_torso_pos_, torso_pos);

    // Define a forward/back Bezier for the FR foot based on the initial torso x-axis.
    const double* torso_mat = data->xmat + 9 * debug_residual_.torso_body_id_;
    double dir[3] = {torso_mat[0], torso_mat[3], 0.0};
    mju_normalize3(dir);

    const int fr = Quadruped::ResidualFn::kFootFR;
    const double* start = initial_foot_pos_[fr];
    double end[3] = {start[0] + kBezierStride * dir[0],
                     start[1] + kBezierStride * dir[1],
                     start[2]};

    mju_copy3(fr_bezier_ctrl_[0], start);
    fr_bezier_ctrl_[1][0] = start[0] + 0.33 * kBezierStride * dir[0];
    fr_bezier_ctrl_[1][1] = start[1] + 0.33 * kBezierStride * dir[1];
    fr_bezier_ctrl_[1][2] = start[2] + 0.08; // Add some height for visualization
    fr_bezier_ctrl_[2][0] = end[0] - 0.33 * kBezierStride * dir[0];
    fr_bezier_ctrl_[2][1] = end[1] - 0.33 * kBezierStride * dir[1];
    fr_bezier_ctrl_[2][2] = start[2] + 0.08;
    mju_copy3(fr_bezier_ctrl_[3], end);

    initialized_ = true;
  }

  // In the interactive app, Task::Transition() is called every step (not env-only),
  // so we must also publish the target + visuals here.
  TransitionEnvOnlyLocked(model, data);
}

void MjTwinDebug::ModifyScene(const mjModel* model, const mjData* data,
                               mjvScene* scene) const {
  // Use the base MjTwin::ModifyScene which calls VisualizeFootholdLogic.
  // We populate foothold_planner_ + foothold_targets inside transitions.
  MjTwin::ModifyScene(model, data, scene);
}

void MjTwinDebug::ResidualFn::Residual(const mjModel* model,
                                       const mjData* data,
                                       double* residual) const {
  if (torso_body_id_ < 0) {
    if (num_residual_ > 0) mju_zero(residual, num_residual_);
    return;
  }
  for (A1Foot foot : kFootAll) {
    if (foot_geom_id_[foot] < 0) {
      if (num_residual_ > 0) mju_zero(residual, num_residual_);
      return;
    }
  }

  int counter = 0;
  double* foot_pos[kNumFoot];
  for (A1Foot foot : kFootAll) {
    foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];
  }

  double avg_foot_pos[3];
  AverageFootPos(avg_foot_pos, foot_pos);

  double* compos = SensorByName(model, data, "torso_subtreecom");
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double* torso_pos = data->xipos + 3 * torso_body_id_;
  const bool is_biped = current_mode_ == kModeBiped;
  const double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;

  const auto* task = static_cast<const MjTwinDebug*>(task_);
  counter = UprightCost(data, residual, counter);
  if (!task || !task->initialized_) {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  } else {
    residual[counter++] = torso_pos[0] - task->initial_torso_pos_[0];
    residual[counter++] = torso_pos[1] - task->initial_torso_pos_[1];
    residual[counter++] = torso_pos[2] - task->initial_torso_pos_[2];
  }

  if (!task || !task->initialized_) {
    for (int i = 0; i < kNumFoot; ++i) {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    }
  } else {
    // Use the same published target that visualization uses (foothold_targets sensor).
    const double* fr_target = nullptr;
    double fr_target_fallback[3] = {0.0, 0.0, 0.0};
    if (double* targets = SensorByName(model, data, "foothold_targets")) {
      fr_target = targets + 3 * kFootFR;
    } else {
      const int gait = static_cast<int>(GetGait());
      if (gait >= 0 && gait < kNumGait) {
        const double phase = GetPhase(data->time);
        const double footphase = 2 * mjPI * kGaitPhase[gait][kFootFR];
        const double t = 0.5 * (1.0 - std::cos(phase - footphase));
        EvalBezier(task->fr_bezier_ctrl_, t, fr_target_fallback);
        fr_target = fr_target_fallback;
      } else {
        fr_target = fr_target_fallback;
      }
    }

    for (A1Foot foot : kFootAll) {
      if (foot == kFootFR) {
        residual[counter++] = foot_pos[foot][0] - fr_target[0];
        residual[counter++] = foot_pos[foot][1] - fr_target[1];
        residual[counter++] = foot_pos[foot][2] - fr_target[2];
      } else {
        residual[counter++] = foot_pos[foot][0] - task->initial_foot_pos_[foot][0];
        residual[counter++] = foot_pos[foot][1] - task->initial_foot_pos_[foot][1];
        residual[counter++] = foot_pos[foot][2] - task->initial_foot_pos_[foot][2];
      }
    }
  }

  if (compos && comvel) {
    counter = BalanceCost(model, data, height_goal, avg_foot_pos, compos, residual, counter);
  } else {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }
  counter = EffortCost(model, data, residual, counter);
  counter = PostureCost(model, data, residual, counter);

  CheckSensorDim(model, counter);
}

}  // namespace mjpc
