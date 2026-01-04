#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/quadruped/mjTwin.h"

#include <string>
#include <cmath>
#include <cstdio>

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc {

void Quadruped::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  // start counter
  int counter = 0;

  // get foot positions
  double* foot_pos[kNumFoot];
  for (A1Foot foot : kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

  // average foot position (height, balance)
  double avg_foot_pos[3];
  AverageFootPos(avg_foot_pos, foot_pos);

  // double* torso_xmat = data->xmat + 9*torso_body_id_;
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  double* compos = SensorByName(model, data, "torso_subtreecom");

  double* torso_pos = data->xipos + 3 * torso_body_id_;             // height, gait(scramble offset), balance
  bool is_biped = current_mode_ == kModeBiped;                      // height, gait(hand filter)
  double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;  // height, balance


  // ---------- Cost Terms ----------
  counter = UprightCost(data, residual, counter);
  // counter = HeightCost(data, torso_pos, height_goal, avg_foot_pos, residual, counter);
  counter = PositionCost(data, goal_pos, residual, counter);
  counter = GaitCost(model, data, torso_pos, is_biped, foot_pos, goal_pos, residual, counter);
  counter = BalanceCost(model, data, height_goal, avg_foot_pos, compos, residual, counter);
  counter = EffortCost(model, data, residual, counter);
  counter = PostureCost(model, data, residual, counter);
  // counter = YawCost(model, data, torso_xmat, residual, counter);        
  // counter = AngularMomentumCost(model, data, residual, counter);
  
  // counter = ClearanceCost(model, data, residual, counter);

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

//  ============  transition  ============
void Quadruped::TransitionLocked(mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
  if (data->time < residual_.last_transition_time_ ||
      residual_.last_transition_time_ == -1) {
    if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
      mode = ResidualFn::kModeScramble;
    }
    residual_.last_transition_time_ = residual_.phase_start_time_ =
        residual_.phase_start_ = data->time;
  }

  // ---------- handle phase velocity change ----------
  double phase_velocity = 2 * mjPI * parameters[residual_.cadence_param_id_];
  if (phase_velocity != residual_.phase_velocity_) {
    residual_.phase_start_ = residual_.GetPhase(data->time);
    residual_.phase_start_time_ = data->time;
    residual_.phase_velocity_ = phase_velocity;
  }


  // ---------- automatic gait switching ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double beta = mju_exp(-(data->time - residual_.last_transition_time_) /
                        ResidualFn::kAutoGaitFilter);
  residual_.com_vel_[0] = beta * residual_.com_vel_[0] + (1 - beta) * comvel[0];
  residual_.com_vel_[1] = beta * residual_.com_vel_[1] + (1 - beta) * comvel[1];
  // TODO(b/268398978): remove reinterpret, int64_t business
  int auto_switch =
      ReinterpretAsInt(parameters[residual_.gait_switch_param_id_]);
  if (mode == ResidualFn::kModeBiped) {
    // biped always trots
    parameters[residual_.gait_param_id_] =
        ReinterpretAsDouble(ResidualFn::kGaitTrot);
  } else if (auto_switch) {
    double com_speed = mju_norm(residual_.com_vel_, 2);
    for (int64_t gait : ResidualFn::kGaitAll) {
      // scramble requires a non-static gait
      if (mode == ResidualFn::kModeScramble && gait == ResidualFn::kGaitStand)
        continue;
      bool lower = com_speed > ResidualFn::kGaitAuto[gait];
      bool upper = gait == ResidualFn::kGaitGallop ||
                   com_speed <= ResidualFn::kGaitAuto[gait + 1];
      bool wait = mju_abs(residual_.gait_switch_time_ - data->time) >
                  ResidualFn::kAutoGaitMinTime;
      if (lower && upper && wait) {
        parameters[residual_.gait_param_id_] = ReinterpretAsDouble(gait);
        residual_.gait_switch_time_ = data->time;
      }
    }
  }


  // ---------- handle gait switch, manual or auto ----------
  double gait_selection = parameters[residual_.gait_param_id_];
  if (gait_selection != residual_.current_gait_) {
    residual_.current_gait_ = gait_selection;
    ResidualFn::A1Gait gait = residual_.GetGait();
    parameters[residual_.duty_param_id_] = ResidualFn::kGaitParam[gait][0];
    parameters[residual_.cadence_param_id_] = ResidualFn::kGaitParam[gait][1];
    parameters[residual_.amplitude_param_id_] = ResidualFn::kGaitParam[gait][2];
    weight[residual_.balance_cost_id_] = ResidualFn::kGaitParam[gait][3];
    weight[residual_.upright_cost_id_] = ResidualFn::kGaitParam[gait][4];
    // weight[residual_.height_cost_id_] = ResidualFn::kGaitParam[gait][5];
  }

  // save mode
  residual_.current_mode_ = static_cast<ResidualFn::A1Mode>(mode);
  residual_.last_transition_time_ = data->time;
}

void Quadruped::ResetLocked(const mjModel* model) {

  mode = ResidualFn::kModeScramble;

  // ----------  task identifiers  ----------
  residual_.gait_param_id_        = ParameterIndex(model, "select_Gait");
  residual_.gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
  residual_.biped_type_param_id_  = ParameterIndex(model, "select_Biped type");
  residual_.cadence_param_id_     = ParameterIndex(model, "Cadence");
  residual_.amplitude_param_id_   = ParameterIndex(model, "Amplitude");
  residual_.duty_param_id_        = ParameterIndex(model, "Duty ratio");
  residual_.arm_posture_param_id_ = ParameterIndex(model, "Arm posture");
  residual_.terrain_type_param_id_ = ParameterIndex(model, "select_Terrain Type");

  residual_.balance_cost_id_ = CostTermByName(model, "Balance");
  residual_.upright_cost_id_ = CostTermByName(model, "Upright");
  // residual_.height_cost_id_  = CostTermByName(model, "Height");

  // ----------  model identifiers  ----------
  residual_.torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "trunk");
  if (residual_.torso_body_id_ < 0) mju_error("body 'trunk' not found");

  residual_.head_site_id_ = mj_name2id(model, mjOBJ_SITE, "head");
  if (residual_.head_site_id_ < 0) mju_error("site 'head' not found");

  int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_id < 0) mju_error("body 'goal' not found");

  residual_.goal_mocap_id_ = model->body_mocapid[goal_id];
  if (residual_.goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

  // foot geom ids
  int foot_index = 0;
  for (const char* footname : {"FL", "HL", "FR", "HR"}) {
    int foot_id = mj_name2id(model, mjOBJ_GEOM, footname);
    if (foot_id < 0) mju_error_s("geom '%s' not found", footname);
    residual_.foot_geom_id_[foot_index] = foot_id;
    foot_index++;
  }

  // shoulder body ids
  int shoulder_index = 0;
  for (const char* shouldername : {"FL_hip", "HL_hip", "FR_hip", "HR_hip"}) {
    int shoulder_id = mj_name2id(model, mjOBJ_BODY, shouldername);
    if (shoulder_id < 0) mju_error_s("body '%s' not found", shouldername);
    residual_.shoulder_body_id_[shoulder_index] = shoulder_id;
    shoulder_index++;
  }

  // knee body ids
  int knee_index = 0;
  for (const char* kneename : {"FL_calf", "HL_calf", "FR_calf", "HR_calf"}) {
    int knee_id = mj_name2id(model, mjOBJ_BODY, kneename);
    if (knee_id < 0) mju_error_s("body '%s' not found", kneename);
    residual_.knee_body_id_[knee_index] = knee_id;
    knee_index++;
  }
  
  // shoulder collision boxes
  const char* shoulder_box_body[ResidualFn::kNumFoot] = {"box_FL_hip_cyl", "box_HL_hip_cyl",
                                                         "box_FR_hip_cyl", "box_HR_hip_cyl"};
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    int bid = mj_name2id(model, mjOBJ_XBODY, shoulder_box_body[foot]);
    if (bid < 0) mju_error_s("body '%s' not found", shoulder_box_body[foot]);
    int mid = model->body_mocapid[bid];
    if (mid < 0) mju_error_s("body '%s' is not mocap", shoulder_box_body[foot]);
    residual_.shoulder_box_mocap_id_[foot] = mid;
  }

  // knee collision boxes
  const char* knee_box_body[ResidualFn::kNumFoot] = {"box_FL_calf_cyl1", "box_HL_calf_cyl1",
                                                     "box_FR_calf_cyl1", "box_HR_calf_cyl1"};
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    int bid = mj_name2id(model, mjOBJ_XBODY, knee_box_body[foot]);
    if (bid < 0) mju_error_s("body '%s' not found", knee_box_body[foot]);
    int mid = model->body_mocapid[bid];
    if (mid < 0) mju_error_s("body '%s' is not mocap", knee_box_body[foot]);
    residual_.knee_box_mocap_id_[foot] = mid;
  }

  // lidar geom id
  int lidar_id = mj_name2id(model, mjOBJ_GEOM, "lidar");
  if (lidar_id < 0) mju_error("geom 'lidar' not found");
  residual_.lidar_geom_id_ = lidar_id;

  // lidar collision box
  int lidar_box_body_id = mj_name2id(model, mjOBJ_XBODY, "box_lidar");
  if (lidar_box_body_id < 0) mju_error("body 'box_lidar' not found");
  int lidar_mid = model->body_mocapid[lidar_box_body_id];
  if (lidar_mid < 0) mju_error("body 'box_lidar' is not mocap");
  residual_.lidar_box_mocap_id_ = lidar_mid;

  // terrain (filled by mjTwin)
  residual_.terrain_ = nullptr;
}

// compute average foot position, depending on mode
void Quadruped::ResidualFn::AverageFootPos(
    double avg_foot_pos[3], double* foot_pos[kNumFoot]) const {
  if (current_mode_ == kModeBiped) {
    int handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
    if (handstand) {
      mju_add3(avg_foot_pos, foot_pos[kFootFL], foot_pos[kFootFR]);
    } else {
      mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    }
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
  } else {
    mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFL]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFR]);
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.25);
  }
}

}  // namespace mjpc
