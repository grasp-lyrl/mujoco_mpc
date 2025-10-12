// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/quadruped/quadruped.h"

#include <string>
#include <cmath>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string QuadrupedHill::XmlPath() const {
  return GetModelPath("quadruped/task_hill.xml");
}
std::string QuadrupedFlat::XmlPath() const {
  return GetModelPath("quadruped/task_flat.xml");
}
std::string QuadrupedHill::Name() const { return "Quadruped Hill"; }
std::string QuadrupedFlat::Name() const { return "Quadruped Flat"; }


// Copy of QuadrupedFlat but with pose targets and keyframe stepping
std::string QuadrupedPose::XmlPath() const {
  return GetModelPath("quadruped/task_pose.xml");
}
std::string QuadrupedPose::Name() const { return "Quadruped Pose"; }

void QuadrupedFlat::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  // start counter
  int counter = 0;

  // get foot positions
  double* foot_pos[kNumFoot];
  for (A1Foot foot : kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

  // average foot position
  double avg_foot_pos[3];
  AverageFootPos(avg_foot_pos, foot_pos);

  double* torso_xmat = data->xmat + 9*torso_body_id_;
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  double* compos = SensorByName(model, data, "torso_subtreecom");


  // ---------- Upright ----------
  if (current_mode_ != kModeFlip) {
    if (current_mode_ == kModeBiped) {
      double biped_type = parameters_[biped_type_param_id_];
      int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
      residual[counter++] = torso_xmat[6] - handstand;
    } else {
      residual[counter++] = torso_xmat[8] - 1;
    }
    residual[counter++] = 0;
    residual[counter++] = 0;
  } else {
    // special handling of flip orientation
    double flip_time = data->time - mode_start_time_;
    double quat[4];
    FlipQuat(quat, flip_time);
    double* torso_xquat = data->xquat + 4*torso_body_id_;
    mju_subQuat(residual + counter, torso_xquat, quat);
    counter += 3;
  }


  // ---------- Height ----------
  // quadrupedal or bipedal height of torso over feet
  double* torso_pos = data->xipos + 3*torso_body_id_;
  bool is_biped = current_mode_ == kModeBiped;
  double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
  if (current_mode_ == kModeScramble) {
    // disable height term in Scramble
    residual[counter++] = 0;
  } else if (current_mode_ == kModeFlip) {
    // height target for Backflip
    double flip_time = data->time - mode_start_time_;
    residual[counter++] = torso_pos[2] - FlipHeight(flip_time);
  } else {
    residual[counter++] = (torso_pos[2] - avg_foot_pos[2]) - height_goal;
  }


  // ---------- Position ----------
  double* head = data->site_xpos + 3*head_site_id_;
  double target[3];
  if (current_mode_ == kModeWalk) {
    // follow prescribed Walk trajectory
    double mode_time = data->time - mode_start_time_;
    Walk(target, mode_time);
  } else {
    // go to the goal mocap body
    target[0] = goal_pos[0];
    target[1] = goal_pos[1];
    target[2] = goal_pos[2];
  }
  residual[counter++] = head[0] - target[0];
  residual[counter++] = head[1] - target[1];
  residual[counter++] =
      current_mode_ == kModeScramble ? 2 * (head[2] - target[2]) : 0;

  // ---------- Gait ----------
  A1Gait gait = GetGait();
  double step[kNumFoot];
  FootStep(step, GetPhase(data->time), gait);
  for (A1Foot foot : kFootAll) {
    if (is_biped) {
      // ignore "hands" in biped mode
      bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
      bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
      bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);
      if (front_hand || back_hand) {
        residual[counter++] = 0;
        continue;
      }
    }
    double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};

    if (current_mode_ == kModeScramble) {
      double torso_to_goal[3];
      double* goal = data->mocap_pos + 3*goal_mocap_id_;
      mju_sub3(torso_to_goal, goal, torso_pos);
      mju_normalize3(torso_to_goal);
      mju_sub3(torso_to_goal, goal, foot_pos[foot]);
      torso_to_goal[2] = 0;
      mju_normalize3(torso_to_goal);
      mju_addToScl3(query, torso_to_goal, 0.15);
    }

    double ground_height = Ground(model, data, query);
    double height_target = ground_height + kFootRadius + step[foot];
    double height_difference = foot_pos[foot][2] - height_target;
    if (current_mode_ == kModeScramble) {
      // in Scramble, foot higher than target is not penalized
      height_difference = mju_min(0, height_difference);
    }
    residual[counter++] = step[foot] ? height_difference : 0;
  }


  // ---------- Balance ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double capture_point[3];
  double fall_time = mju_sqrt(2*height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];


  // ---------- Effort ----------
  mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
  counter += model->nu;


  // ---------- Posture ----------
  double* home = KeyQPosByName(model, data, "home");
  mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
  if (current_mode_ == kModeFlip) {
    double flip_time = data->time - mode_start_time_;
    if (flip_time < crouch_time_) {
      double* crouch = KeyQPosByName(model, data, "crouch");
      mju_sub(residual + counter, data->qpos + 7, crouch + 7, model->nu);
    } else if (flip_time >= crouch_time_ &&
               flip_time < jump_time_ + flight_time_) {
      // free legs during flight phase
      mju_zero(residual + counter, model->nu);
    }
  }
  for (A1Foot foot : kFootAll) {
    for (int joint = 0; joint < 3; joint++) {
      residual[counter + 3*foot + joint] *= kJointPostureGain[joint];
    }
  }
  if (current_mode_ == kModeBiped) {
    // loosen the "hands" in Biped mode
    bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
    double arm_posture = parameters_[arm_posture_param_id_];
    if (handstand) {
      residual[counter + 6] *= arm_posture;
      residual[counter + 7] *= arm_posture;
      residual[counter + 8] *= arm_posture;
      residual[counter + 9] *= arm_posture;
      residual[counter + 10] *= arm_posture;
      residual[counter + 11] *= arm_posture;
    } else {
      residual[counter + 0] *= arm_posture;
      residual[counter + 1] *= arm_posture;
      residual[counter + 2] *= arm_posture;
      residual[counter + 3] *= arm_posture;
      residual[counter + 4] *= arm_posture;
      residual[counter + 5] *= arm_posture;
    }
  }
  counter += model->nu;


  // ---------- Yaw ----------
  double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
  if (current_mode_ == kModeBiped) {
    int handstand =
        ReinterpretAsInt(parameters_[biped_type_param_id_]) ? 1 : -1;
    torso_heading[0] = handstand * torso_xmat[2];
    torso_heading[1] = handstand * torso_xmat[5];
  }
  mju_normalize(torso_heading, 2);
  double heading_goal = parameters_[ParameterIndex(model, "Heading")];
  residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
  residual[counter++] = torso_heading[1] - mju_sin(heading_goal);


  // ---------- Angular momentum ----------
  mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));
  counter +=3;

  // ---------- FootCost (CPU-side cost map) ----------
  if (foot_cost_id_ >= 0) {
    for (A1Foot foot : kFootAll) {
      double sample = 0.0;
      const double* foot_p = data->geom_xpos + 3 * foot_geom_id_[foot];

      // stance gating using terrain hfield (continuous), optional
      double stance_w = 1.0;
      if (terrain_geom_id_ >= 0 && terrain_hfield_id_ >= 0) {
        const double* tpos = model->geom_pos + 3 * terrain_geom_id_;
        double tx = foot_p[0] - tpos[0];
        double ty = foot_p[1] - tpos[1];
        const double* thf = model->hfield_size + 4 * terrain_hfield_id_;
        double sx_t = thf[0], sy_t = thf[1];
        double ground_h = foot_p[2];
        if (mju_abs(tx) <= sx_t && mju_abs(ty) <= sy_t) {
          int nrow_t = model->hfield_nrow[terrain_hfield_id_];
          int ncol_t = model->hfield_ncol[terrain_hfield_id_];
          int adr_t = model->hfield_adr[terrain_hfield_id_];
          double u_t = (tx / (2.0 * sx_t)) + 0.5;
          double v_t = (ty / (2.0 * sy_t)) + 0.5;
          double xt = mju_clip(u_t * (ncol_t - 1), 0.0, (double)(ncol_t - 1));
          double yt = mju_clip(v_t * (nrow_t - 1), 0.0, (double)(nrow_t - 1));
          int x0t = (int) mju_floor(xt);
          int y0t = (int) mju_floor(yt);
          int x1t = mju_min(x0t + 1, ncol_t - 1);
          int y1t = mju_min(y0t + 1, nrow_t - 1);
          double txt = xt - x0t;
          double tyt = yt - y0t;
          const float* data_t = model->hfield_data + adr_t;
          double w00 = data_t[y0t * ncol_t + x0t];
          double w10 = data_t[y0t * ncol_t + x1t];
          double w01 = data_t[y1t * ncol_t + x0t];
          double w11 = data_t[y1t * ncol_t + x1t];
          double w0 = (1.0 - txt) * w00 + txt * w10;
          double w1 = (1.0 - txt) * w01 + txt * w11;
          ground_h = tpos[2] + thf[2] * ((1.0 - tyt) * w0 + tyt * w1);
        }
        double dz = mju_max(0.0, foot_p[2] - ground_h);
        stance_w = mju_clip(1.0 - dz / (2.0 * ResidualFn::kFootRadius), 0.0, 1.0);
      }

      // bilinear sample CPU cost map
      const auto& cm = cost_map_;
      if (cm.width > 0 && cm.height > 0 && !cm.data.empty()) {
        double fx = (foot_p[0] - cm.origin_x) / cm.resolution;
        double fy = (foot_p[1] - cm.origin_y) / cm.resolution;
        int x0 = (int) mju_floor(fx);
        int y0 = (int) mju_floor(fy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        if (x0 >= 0 && y0 >= 0 && x1 < cm.width && y1 < cm.height) {
          double tx = fx - x0;
          double ty = fy - y0;
          const float* grid = cm.data.data();
          float v00 = grid[y0 * cm.width + x0];
          float v10 = grid[y0 * cm.width + x1];
          float v01 = grid[y1 * cm.width + x0];
          float v11 = grid[y1 * cm.width + x1];
          double v0 = (1.0 - tx) * v00 + tx * v10;
          double v1 = (1.0 - tx) * v01 + tx * v11;
          sample = ((1.0 - ty) * v0 + ty * v1) * stance_w;
        }
      }

      // Residual shaping: make stage cost linear in sample by emitting sqrt(sample + eps)
      constexpr double eps = 1e-12;
      residual[counter++] = mju_sqrt(mju_max(0.0, sample) + eps);
    }
  }

  // ---------- Terrain-normal clearance cost (optional) ----------
  if (clear_cost_id_ >= 0) {
    // Only compute for mjTwin; otherwise emit zeros (match FootCost pattern)
    auto twin = dynamic_cast<const MjTwin*>(task_);
    if (twin && terrain_geom_id_ >= 0) {
      constexpr double beta = 60.0;
      // knees: FL, FR, HL, HR (immaterial spheres centered at calf COM)
      for (int k = 0; k < 4; ++k) {
        int bid = knee_body_id_clear_[k];
        if (bid >= 0) {
          const double* pk = data->xpos + 3 * bid;
          double s_world[3], n_world[3];
          if (twin->TerrainSurfaceAndNormalWorld(model, data, pk[0], pk[1], s_world, n_world)) {
            double p_minus_s[3] = {pk[0]-s_world[0], pk[1]-s_world[1], pk[2]-s_world[2]};
          double sN = mju_dot(n_world, p_minus_s, 3) - knee_radius_clear_;
          constexpr double margin = 0.05;
          double u = std::log1p(mju_exp(beta * (margin - sN))) / beta;
            residual[counter++] = u;
          } else {
            residual[counter++] = 0;
          }
        } else {
          residual[counter++] = 0;
        }
      }

      // trunk cylinder geom (treated as sphere with radius=size[0] at geom center)
      if (trunk_cyl_geom_id_clear_ >= 0) {
        const double* pl = data->geom_xpos + 3 * trunk_cyl_geom_id_clear_;
        double s_world[3], n_world[3];
        if (twin->TerrainSurfaceAndNormalWorld(model, data, pl[0], pl[1], s_world, n_world)) {
          double p_minus_s[3] = {pl[0]-s_world[0], pl[1]-s_world[1], pl[2]-s_world[2]};
          double sN = mju_dot(n_world, p_minus_s, 3) - trunk_cyl_radius_clear_;
          constexpr double margin = 0.10;
          double u = std::log1p(mju_exp(beta * (margin - sN))) / beta;
          residual[counter++] = u;
        } else {
          residual[counter++] = 0;
        }
      } else {
        residual[counter++] = 0;
      }

      // trunk sphere geom
      if (trunk_sph_geom_id_clear_ >= 0) {
        const double* pl = data->geom_xpos + 3 * trunk_sph_geom_id_clear_;
        double s_world[3], n_world[3];
        if (twin->TerrainSurfaceAndNormalWorld(model, data, pl[0], pl[1], s_world, n_world)) {
          double p_minus_s[3] = {pl[0]-s_world[0], pl[1]-s_world[1], pl[2]-s_world[2]};
          double sN = mju_dot(n_world, p_minus_s, 3) - trunk_sph_radius_clear_;
          constexpr double margin = 0.10;
          double u = std::log1p(mju_exp(beta * (margin - sN))) / beta;
          residual[counter++] = u;
        } else {
          residual[counter++] = 0;
        }
      } else {
        residual[counter++] = 0;
      }
    } else {
      // Non-mjTwin tasks or missing terrain: append zeros to match dims (head + 4 knees + lidar = 6)
      mju_zero(residual + counter, 6);
      counter += 6;
    }
  }


  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

//  ============  transition  ============
void QuadrupedFlat::TransitionLocked(mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
  if (data->time < residual_.last_transition_time_ ||
      residual_.last_transition_time_ == -1) {
    if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
      mode = ResidualFn::kModeQuadruped;  // mode stateful, switch to Quadruped
    }
    residual_.last_transition_time_ = residual_.phase_start_time_ =
        residual_.phase_start_ = data->time;
  }

  // ---------- prevent forbidden mode transitions ----------
  // switching mode, not from quadruped
  if (mode != residual_.current_mode_ &&
      residual_.current_mode_ != ResidualFn::kModeQuadruped) {
    // switch into stateful mode only allowed from Quadruped
    if (mode == ResidualFn::kModeWalk || mode == ResidualFn::kModeFlip) {
      mode = ResidualFn::kModeQuadruped;
    }
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
    weight[residual_.height_cost_id_] = ResidualFn::kGaitParam[gait][5];
  }


  // ---------- Walk ----------
  double* goal_pos = data->mocap_pos + 3*residual_.goal_mocap_id_;
  if (mode == ResidualFn::kModeWalk) {
    double angvel = parameters[ParameterIndex(model, "Walk turn")];
    double speed = parameters[ParameterIndex(model, "Walk speed")];

    // current torso direction
    double* torso_xmat = data->xmat + 9*residual_.torso_body_id_;
    double forward[2] = {torso_xmat[0], torso_xmat[3]};
    mju_normalize(forward, 2);
    double leftward[2] = {-forward[1], forward[0]};

    // switching into Walk or parameters changed, reset task state
    if (mode != residual_.current_mode_ || residual_.angvel_ != angvel ||
        residual_.speed_ != speed) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save current speed and angvel
      residual_.speed_ = speed;
      residual_.angvel_ = angvel;

      // compute and save rotation axis / walk origin
      double axis[2] = {data->xpos[3*residual_.torso_body_id_],
                        data->xpos[3*residual_.torso_body_id_+1]};
      if (mju_abs(angvel) > ResidualFn::kMinAngvel) {
        // don't allow turning with very small angvel
        double d = speed / angvel;
        axis[0] += d * leftward[0];
        axis[1] += d * leftward[1];
      }
      residual_.position_[0] = axis[0];
      residual_.position_[1] = axis[1];

      // save vector from axis to initial goal position
      residual_.heading_[0] = goal_pos[0] - axis[0];
      residual_.heading_[1] = goal_pos[1] - axis[1];
    }

    // move goal
    double time = data->time - residual_.mode_start_time_;
    residual_.Walk(goal_pos, time);
  }


  // ---------- Flip ----------
  double* compos = SensorByName(model, data, "torso_subtreecom");
  if (mode == ResidualFn::kModeFlip) {
    // switching into Flip, reset task state
    if (mode != residual_.current_mode_) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save body orientation, ground height
      mju_copy4(residual_.orientation_,
                data->xquat + 4 * residual_.torso_body_id_);
      residual_.ground_ = Ground(model, data, compos);

      // save parameters
      residual_.save_weight_ = weight;
      residual_.save_gait_switch_ = parameters[residual_.gait_switch_param_id_];

      // set parameters
      weight[CostTermByName(model, "Upright")] = 0.2;
      weight[CostTermByName(model, "Height")] = 5;
      weight[CostTermByName(model, "Position")] = 0;
      weight[CostTermByName(model, "Gait")] = 0;
      weight[CostTermByName(model, "Balance")] = 0;
      weight[CostTermByName(model, "Effort")] = 0.005;
      weight[CostTermByName(model, "Posture")] = 0.1;
      parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(0);
    }

    // time from start of Flip
    double flip_time = data->time - residual_.mode_start_time_;

    if (flip_time >=
        residual_.jump_time_ + residual_.flight_time_ + residual_.land_time_) {
      // Flip ended, back to Quadruped, restore values
      mode = ResidualFn::kModeQuadruped;
      weight = residual_.save_weight_;
      parameters[residual_.gait_switch_param_id_] = residual_.save_gait_switch_;
      goal_pos[0] = data->site_xpos[3*residual_.head_site_id_ + 0];
      goal_pos[1] = data->site_xpos[3*residual_.head_site_id_ + 1];
    }
  }

  // save mode
  residual_.current_mode_ = static_cast<ResidualFn::A1Mode>(
      mjMIN(static_cast<int>(ResidualFn::kNumMode)-1, mjMAX(0, mode)));
  residual_.last_transition_time_ = data->time;
}

// colors of visualisation elements drawn in ModifyScene()
constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};  // step-height cylinders
constexpr float kHullRgba[4] = {0.4, 0.2, 0.8, 1};  // convex hull
constexpr float kAvgRgba[4] = {0.4, 0.2, 0.8, 1};   // average foot position
constexpr float kCapRgba[4] = {0.3, 0.3, 0.8, 1};   // capture point
constexpr float kPcpRgba[4] = {0.5, 0.5, 0.2, 1};   // projected capture point

// draw task-related geometry in the scene
void QuadrupedFlat::ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {
  // flip target pose
  if (residual_.current_mode_ == ResidualFn::kModeFlip) {
    double flip_time = data->time - residual_.mode_start_time_;
    double* torso_pos = data->xpos + 3*residual_.torso_body_id_;
    double pos[3] = {torso_pos[0], torso_pos[1],
                     residual_.FlipHeight(flip_time)};
    double quat[4];
    residual_.FlipQuat(quat, flip_time);
    double mat[9];
    mju_quat2Mat(mat, quat);
    double size[3] = {0.25, 0.15, 0.05};
    float rgba[4] = {0, 1, 0, 0.5};
    AddGeom(scene, mjGEOM_BOX, size, pos, mat, rgba);

    // don't draw anything else during flip
    return;
  }

  // current foot positions
  double* foot_pos[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * residual_.foot_geom_id_[foot];

  // stance and flight positions
  double flight_pos[ResidualFn::kNumFoot][3];
  double stance_pos[ResidualFn::kNumFoot][3];
  // set to foot horizontal position:
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    flight_pos[foot][0] = stance_pos[foot][0] = foot_pos[foot][0];
    flight_pos[foot][1] = stance_pos[foot][1] = foot_pos[foot][1];
  }

  // ground height below feet
  double ground[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    ground[foot] = Ground(model, data, foot_pos[foot]);
  }

  // step heights
  ResidualFn::A1Gait gait = residual_.GetGait();
  double step[ResidualFn::kNumFoot];
  residual_.FootStep(step, residual_.GetPhase(data->time), gait);

  // draw step height
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    stance_pos[foot][2] = ResidualFn::kFootRadius + ground[foot];
    if (residual_.current_mode_ == ResidualFn::kModeBiped) {
      // skip "hands" in biped mode
      bool handstand =
          ReinterpretAsInt(parameters[residual_.biped_type_param_id_]);
      bool front_hand = !handstand && (foot == ResidualFn::kFootFL ||
                                       foot == ResidualFn::kFootFR);
      bool back_hand = handstand && (foot == ResidualFn::kFootHL ||
                                     foot == ResidualFn::kFootHR);
      if (front_hand || back_hand) continue;
    }
    if (step[foot]) {
      flight_pos[foot][2] = ResidualFn::kFootRadius + step[foot] + ground[foot];
      AddConnector(scene, mjGEOM_CYLINDER, ResidualFn::kFootRadius,
                   stance_pos[foot], flight_pos[foot], kStepRgba);
    }
  }

  // support polygon (currently unused for cost)
  double polygon[2*ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    polygon[2*foot] = foot_pos[foot][0];
    polygon[2*foot + 1] = foot_pos[foot][1];
  }
  int hull[ResidualFn::kNumFoot];
  int num_hull = Hull2D(hull, ResidualFn::kNumFoot, polygon);
  for (int i=0; i < num_hull; i++) {
    int j = (i + 1) % num_hull;
    AddConnector(scene, mjGEOM_CAPSULE, ResidualFn::kFootRadius/2,
                 stance_pos[hull[i]], stance_pos[hull[j]], kHullRgba);
  }

  // capture point
  bool is_biped = residual_.current_mode_ == ResidualFn::kModeBiped;
  double height_goal =
      is_biped ? ResidualFn::kHeightBiped : ResidualFn::kHeightQuadruped;
  double fall_time = mju_sqrt(2*height_goal / residual_.gravity_);
  double capture[3];
  double* compos = SensorByName(model, data, "torso_subtreecom");
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  mju_addScl3(capture, compos, comvel, fall_time);

  // ground under CoM
  double com_ground = Ground(model, data, compos);

  // average foot position
  double feet_pos[3];
  residual_.AverageFootPos(feet_pos, foot_pos);
  feet_pos[2] = com_ground;

  double foot_size[3] = {ResidualFn::kFootRadius, 0, 0};

  // average foot position
  AddGeom(scene, mjGEOM_SPHERE, foot_size, feet_pos, /*mat=*/nullptr, kAvgRgba);

  // capture point
  capture[2] = com_ground;
  AddGeom(scene, mjGEOM_SPHERE, foot_size, capture, /*mat=*/nullptr, kCapRgba);

  // capture point, projected onto hull
  double pcp2[2];
  NearestInHull(pcp2, capture, polygon, hull, num_hull);
  double pcp[3] = {pcp2[0], pcp2[1], com_ground};
  AddGeom(scene, mjGEOM_SPHERE, foot_size, pcp, /*mat=*/nullptr, kPcpRgba);
}

//  ============  task-state utilities  ============
// save task-related ids
void QuadrupedFlat::ResetLocked(const mjModel* model) {
  // ----------  task identifiers  ----------
  residual_.gait_param_id_ = ParameterIndex(model, "select_Gait");
  residual_.gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
  residual_.flip_dir_param_id_ = ParameterIndex(model, "select_Flip dir");
  residual_.biped_type_param_id_ = ParameterIndex(model, "select_Biped type");
  residual_.cadence_param_id_ = ParameterIndex(model, "Cadence");
  residual_.amplitude_param_id_ = ParameterIndex(model, "Amplitude");
  residual_.duty_param_id_ = ParameterIndex(model, "Duty ratio");
  residual_.arm_posture_param_id_ = ParameterIndex(model, "Arm posture");
  residual_.balance_cost_id_ = CostTermByName(model, "Balance");
  residual_.upright_cost_id_ = CostTermByName(model, "Upright");
  residual_.height_cost_id_ = CostTermByName(model, "Height");

  // optional high-res FootCost term (only present in mjTwin)
  residual_.foot_cost_id_ = CostTermByName(model, "FootCost");

  // clearance cost term id (optional; user sensor named "NormClear")
  residual_.clear_cost_id_ = CostTermByName(model, "NormClear");
  // default: zero weight unless explicitly enabled (avoid affecting other tasks)
  if (residual_.clear_cost_id_ >= 0) {
    // set weight to 0 for all tasks by default; will set non-zero in MjTwin Reset
    weight[residual_.clear_cost_id_] = 0.0;
  }

  // Initialize current gait from parameter selection to avoid an immediate
  // gait-change override on the first Transition call.
  if (residual_.gait_param_id_ >= 0) {
    residual_.current_gait_ = parameters[residual_.gait_param_id_];
  }

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
  // optional: cache ids for costmap assets if present
  residual_.cost_hfield_id_ = mj_name2id(model, mjOBJ_HFIELD, "costmap");
  residual_.cost_geom_id_ = mj_name2id(model, mjOBJ_GEOM, "terrain_cost");
  // also cache terrain ids for stance gating via ground query
  residual_.terrain_hfield_id_ = mj_name2id(model, mjOBJ_HFIELD, "hf133");
  residual_.terrain_geom_id_ = mj_name2id(model, mjOBJ_GEOM, "terrain");

  // clearance sites: head site and calf bodies
  residual_.head_site_id_clear_ = mj_name2id(model, mjOBJ_SITE, "head");
  // identify the two forward trunk collision geoms: cylinder and sphere
  // find the named head and lidar geoms explicitly
  residual_.trunk_cyl_geom_id_clear_ = mj_name2id(model, mjOBJ_GEOM, "head_cyl");
  if (residual_.trunk_cyl_geom_id_clear_ >= 0) {
    residual_.trunk_cyl_radius_clear_ = model->geom_size[3 * residual_.trunk_cyl_geom_id_clear_ + 0];
  }
  residual_.trunk_sph_geom_id_clear_ = mj_name2id(model, mjOBJ_GEOM, "lidar_sph");
  if (residual_.trunk_sph_geom_id_clear_ >= 0) {
    residual_.trunk_sph_radius_clear_ = model->geom_size[3 * residual_.trunk_sph_geom_id_clear_ + 0];
  }
  residual_.knee_body_id_clear_[0] = mj_name2id(model, mjOBJ_BODY, "FL_calf");
  residual_.knee_body_id_clear_[1] = mj_name2id(model, mjOBJ_BODY, "FR_calf");
  residual_.knee_body_id_clear_[2] = mj_name2id(model, mjOBJ_BODY, "HL_calf");
  residual_.knee_body_id_clear_[3] = mj_name2id(model, mjOBJ_BODY, "HR_calf");

  // radii for clearance cost (match visualization defaults); override head with collision sphere if found
  residual_.knee_radius_clear_ = 0.03;
  residual_.head_radius_clear_ = 0.03;
  int trunk_bid = mj_name2id(model, mjOBJ_BODY, "trunk");
  if (trunk_bid >= 0) {
    int best = -1;
    double bestd2 = 1e30;
    for (int gi = 0; gi < model->ngeom; ++gi) {
      if (model->geom_type[gi] != mjGEOM_SPHERE) continue;
      if (model->geom_bodyid[gi] != trunk_bid) continue;
      if (model->geom_group[gi] != 3) continue;
      double dx = model->geom_pos[3*gi+0];
      double dy = model->geom_pos[3*gi+1];
      double dz = model->geom_pos[3*gi+2];
      double d2 = dx*dx + dy*dy + dz*dz;
      if (d2 < bestd2) { bestd2 = d2; best = gi; }
    }
    if (best >= 0) residual_.head_radius_clear_ = model->geom_size[3 * best + 0];
  }

  // Initialize CPU-side FootCost map from XML hfield (visual) if present.
  // This provides a default cost that external frameworks can later override.
  if ((residual_.cost_map_.width == 0 || residual_.cost_map_.height == 0 ||
       residual_.cost_map_.data.empty()) &&
      residual_.cost_hfield_id_ >= 0 && residual_.cost_geom_id_ >= 0) {
    int hid = residual_.cost_hfield_id_;
    int nrow = model->hfield_nrow[hid];
    int ncol = model->hfield_ncol[hid];
    int adr  = model->hfield_adr[hid];
    if (nrow > 0 && ncol > 0) {
      const double* gpos  = model->geom_pos + 3 * residual_.cost_geom_id_;
      const double* hsize = model->hfield_size + 4 * hid;  // [sx, sy, sz, ...]
      double sx = hsize[0];
      double sy = hsize[1];
      // Match MuJoCo's bilinear sampling mapping: xt in [0..ncol-1] spans 2*sx.
      double resx = (ncol > 1) ? (2.0 * sx) / (ncol - 1) : (2.0 * sx);
      // Use square cells; prefer X resolution (images typically square here).
      double resolution = resx;

      residual_.cost_map_.origin_x   = gpos[0] - sx;
      residual_.cost_map_.origin_y   = gpos[1] - sy;
      residual_.cost_map_.resolution = resolution;
      residual_.cost_map_.width      = ncol;
      residual_.cost_map_.height     = nrow;
      size_t count = static_cast<size_t>(nrow) * static_cast<size_t>(ncol);
      const float* src = model->hfield_data + adr;
      residual_.cost_map_.data.assign(src, src + count);
      // bump version so downstream can detect initialization
      residual_.cost_map_.version += 1;
    }
  }

  // Stop zeroing any visual hfield; computation uses CPU-side cost map only.
  int shoulder_index = 0;
  for (const char* shouldername : {"FL_hip", "HL_hip", "FR_hip", "HR_hip"}) {
    int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
    if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
    residual_.shoulder_body_id_[shoulder_index] = foot_id;
    shoulder_index++;
  }

  // ----------  derived kinematic quantities for Flip  ----------
  residual_.gravity_ = mju_norm3(model->opt.gravity);
  // velocity at takeoff
  residual_.jump_vel_ =
      mju_sqrt(2 * residual_.gravity_ *
               (ResidualFn::kMaxHeight - ResidualFn::kLeapHeight));
  // time in flight phase
  residual_.flight_time_ = 2 * residual_.jump_vel_ / residual_.gravity_;
  // acceleration during jump phase
  residual_.jump_acc_ =
      residual_.jump_vel_ * residual_.jump_vel_ /
      (2 * (ResidualFn::kLeapHeight - ResidualFn::kCrouchHeight));
  // time in crouch sub-phase of jump
  residual_.crouch_time_ =
      mju_sqrt(2 * (ResidualFn::kHeightQuadruped - ResidualFn::kCrouchHeight) /
               residual_.jump_acc_);
  // time in leap sub-phase of jump
  residual_.leap_time_ = residual_.jump_vel_ / residual_.jump_acc_;
  // jump total time
  residual_.jump_time_ = residual_.crouch_time_ + residual_.leap_time_;
  // velocity at beginning of crouch
  residual_.crouch_vel_ = -residual_.jump_acc_ * residual_.crouch_time_;
  // time of landing phase
  residual_.land_time_ =
      2 * (ResidualFn::kLeapHeight - ResidualFn::kHeightQuadruped) /
      residual_.jump_vel_;
  // acceleration during landing
  residual_.land_acc_ = residual_.jump_vel_ / residual_.land_time_;
  // rotational velocity during flight phase (rotates 1.25 pi)
  residual_.flight_rot_vel_ = 1.25 * mjPI / residual_.flight_time_;
  // rotational velocity at start of leap (rotates 0.5 pi)
  residual_.jump_rot_vel_ =
      mjPI / residual_.leap_time_ - residual_.flight_rot_vel_;
  // rotational acceleration during leap (rotates 0.5 pi)
  residual_.jump_rot_acc_ =
      (residual_.flight_rot_vel_ - residual_.jump_rot_vel_) /
      residual_.leap_time_;
  // rotational deceleration during land (rotates 0.25 pi)
  residual_.land_rot_acc_ =
      2 * (residual_.flight_rot_vel_ * residual_.land_time_ - mjPI / 4) /
      (residual_.land_time_ * residual_.land_time_);
}

// compute average foot position, depending on mode
void QuadrupedFlat::ResidualFn::AverageFootPos(
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

// return phase as a function of time
double QuadrupedFlat::ResidualFn::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// horizontal Walk trajectory
void QuadrupedFlat::ResidualFn::Walk(double pos[2], double time) const {
  if (mju_abs(angvel_) < kMinAngvel) {
    // no rotation, go in straight line
    double forward[2] = {heading_[0], heading_[1]};
    mju_normalize(forward, 2);
    pos[0] = position_[0] + heading_[0] + time*speed_*forward[0];
    pos[1] = position_[1] + heading_[1] + time*speed_*forward[1];
  } else {
    // walk on a circle
    double angle = time * angvel_;
    double mat[4] = {mju_cos(angle), -mju_sin(angle),
                     mju_sin(angle),  mju_cos(angle)};
    mju_mulMatVec(pos, mat, heading_, 2, 2);
    pos[0] += position_[0];
    pos[1] += position_[1];
  }
}

// get gait
QuadrupedFlat::ResidualFn::A1Gait QuadrupedFlat::ResidualFn::GetGait() const {
  if (current_mode_ == kModeBiped)
    return kGaitTrot;
  return static_cast<A1Gait>(ReinterpretAsInt(current_gait_));
}

// return normalized target step height
double QuadrupedFlat::ResidualFn::StepHeight(double time, double footphase,
                                             double duty_ratio) const {
  double angle = fmod(time + mjPI - footphase, 2*mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

// compute target step height for all feet
void QuadrupedFlat::ResidualFn::FootStep(double step[kNumFoot], double time,
                                         A1Gait gait) const {
  double amplitude = parameters_[amplitude_param_id_];
  double duty_ratio = parameters_[duty_param_id_];
  for (A1Foot foot : kFootAll) {
    double footphase = 2*mjPI*kGaitPhase[gait][foot];
    step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
  }
}

// height during flip
double QuadrupedFlat::ResidualFn::FlipHeight(double time) const {
  if (time >= jump_time_ + flight_time_ + land_time_) {
    return kHeightQuadruped + ground_;
  }
  double h = 0;
  if (time < jump_time_) {
    h = kHeightQuadruped + time * crouch_vel_ + 0.5 * time * time * jump_acc_;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    h = kLeapHeight + jump_vel_*time - 0.5*9.81*time*time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    h = kLeapHeight - jump_vel_*time + 0.5*land_acc_*time*time;
  }
  return h + ground_;
}

// orientation during flip
//  total rotation = leap + flight + land
//            2*pi = pi/2 + 5*pi/4 + pi/4
void QuadrupedFlat::ResidualFn::FlipQuat(double quat[4], double time) const {
  double angle = 0;
  if (time >= jump_time_ + flight_time_ + land_time_) {
    angle = 2*mjPI;
  } else if (time >= crouch_time_ && time < jump_time_) {
    time -= crouch_time_;
    angle = 0.5 * jump_rot_acc_ * time * time + jump_rot_vel_ * time;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    angle = mjPI/2 + flight_rot_vel_ * time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    angle = 1.75*mjPI + flight_rot_vel_*time - 0.5*land_rot_acc_ * time * time;
  }
  int flip_dir = ReinterpretAsInt(parameters_[flip_dir_param_id_]);
  double axis[3] = {0, flip_dir ? 1.0 : -1.0, 0};
  mju_axisAngle2Quat(quat, axis, angle);
  mju_mulQuat(quat, orientation_, quat);
}


// --------------------- Residuals for quadruped task --------------------
//   Number of residuals: 4
//     Residual (0): position_z - average(foot position)_z - height_goal
//     Residual (1): position - goal_position
//     Residual (2): orientation - goal_orientation
//     Residual (3): control
//   Number of parameters: 1
//     Parameter (1): height_goal
// -----------------------------------------------------------------------
void QuadrupedHill::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  // ---------- Residual (0) ----------
  // standing height goal
  double height_goal = parameters_[0];

  // system's standing height
  double standing_height = SensorByName(model, data, "position")[2];

  // average foot height
  double FRz = SensorByName(model, data, "FR")[2];
  double FLz = SensorByName(model, data, "FL")[2];
  double RRz = SensorByName(model, data, "RR")[2];
  double RLz = SensorByName(model, data, "RL")[2];
  double avg_foot_height = 0.25 * (FRz + FLz + RRz + RLz);

  residual[0] = (standing_height - avg_foot_height) - height_goal;

  // ---------- Residual (1) ----------
  // goal position
  const double* goal_position = data->mocap_pos;

  // system's position
  double* position = SensorByName(model, data, "position");

  // position error
  mju_sub3(residual + 1, position, goal_position);

  // ---------- Residual (2) ----------
  // goal orientation
  double goal_rotmat[9];
  const double* goal_orientation = data->mocap_quat;
  mju_quat2Mat(goal_rotmat, goal_orientation);

  // system's orientation
  double body_rotmat[9];
  double* orientation = SensorByName(model, data, "orientation");
  mju_quat2Mat(body_rotmat, orientation);

  mju_sub(residual + 4, body_rotmat, goal_rotmat, 9);

  // ---------- Residual (3) ----------
  mju_copy(residual + 13, data->ctrl, model->nu);
}

// -------- Transition for quadruped task --------
//   If quadruped is within tolerance of goal ->
//   set goal to next from keyframes.
// -----------------------------------------------
void QuadrupedHill::TransitionLocked(mjModel* model, mjData* data) {
  // set mode to GUI selection
  if (mode > 0) {
    residual_.current_mode_ = mode - 1;
  } else {
    // ---------- Compute tolerance ----------
    // goal position
    const double* goal_position = data->mocap_pos;

    // goal orientation
    const double* goal_orientation = data->mocap_quat;

    // system's position
    double* position = SensorByName(model, data, "position");

    // system's orientation
    double* orientation = SensorByName(model, data, "orientation");

    // position error
    double position_error[3];
    mju_sub3(position_error, position, goal_position);
    double position_error_norm = mju_norm3(position_error);

    // orientation error
    double geodesic_distance =
        1.0 - mju_abs(mju_dot(goal_orientation, orientation, 4));

    // ---------- Check tolerance ----------
    double tolerance = 1.5e-1;
    if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
      // update task state
      residual_.current_mode_ += 1;
      if (residual_.current_mode_ == model->nkey) {
        residual_.current_mode_ = 0;
      }
    }
  }

  // ---------- Set goal ----------
  mju_copy3(data->mocap_pos, model->key_mpos + 3 * residual_.current_mode_);
  mju_copy4(data->mocap_quat, model->key_mquat + 4 * residual_.current_mode_);
}


// Copy of QuadrupedFlatPose but with pose targets and keyframe stepping
void QuadrupedPose::ResidualFn::Residual(const mjModel* model,
                                          const mjData* data,
                                          double* residual) const {
   // start counter
   int counter = 0;

  // ---------- Pose (optional) ----------
  // If a pose is selected from UI (select_Pose > 0), add Pose residual first.
  int pose_index = 0;
  if (pose_select_param_id_ >= 0) {
    pose_index = ReinterpretAsInt(parameters_[pose_select_param_id_]);
  }
  // Position-only targets are indices 1..4. Full pose targets are indices >=5.
  bool pose_target_selected = pose_index >= 5;
  if (pose_target_selected && model->nkey >= pose_index + 1) {
    // layout: 3 pos + 9 rotmat difference
    const double* goal_position = model->key_mpos + 3 * pose_index;
    const double* goal_quat = model->key_mquat + 4 * pose_index;
    double goal_rotmat[9];
    mju_quat2Mat(goal_rotmat, goal_quat);

    // system pose from torso body state
    const double* position = data->xpos + 3 * torso_body_id_;
    const double* torso_xquat = data->xquat + 4 * torso_body_id_;
    double body_rotmat[9];
    mju_quat2Mat(body_rotmat, torso_xquat);

    // position residual (3)
    mju_sub3(residual + counter, position, goal_position);
    counter += 3;

    // orientation residual (9)
    mju_sub(residual + counter, body_rotmat, goal_rotmat, 9);
    counter += 9;
  } else {
    // no full pose selected: fill Pose term with zeros to keep dimensions consistent
    mju_zero(residual + counter, 12);
    counter += 12;
  }

   // get foot positions
   double* foot_pos[kNumFoot];
   for (A1Foot foot : kFootAll)
     foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

   // average foot position
   double avg_foot_pos[3];
   AverageFootPos(avg_foot_pos, foot_pos);

   double* torso_xmat = data->xmat + 9*torso_body_id_;
   double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
   double* compos = SensorByName(model, data, "torso_subtreecom");


  // ---------- Upright ----------
   if (current_mode_ != kModeFlip) {
     if (current_mode_ == kModeBiped) {
       double biped_type = parameters_[biped_type_param_id_];
       int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
       residual[counter++] = torso_xmat[6] - handstand;
     } else {
       residual[counter++] = torso_xmat[8] - 1;
     }
     residual[counter++] = 0;
     residual[counter++] = 0;
   } else {
    // special handling of flip orientation
     double flip_time = data->time - mode_start_time_;
     double quat[4];
     FlipQuat(quat, flip_time);
     double* torso_xquat = data->xquat + 4*torso_body_id_;
     mju_subQuat(residual + counter, torso_xquat, quat);
     counter += 3;
   }


  // ---------- Height ----------
  // quadrupedal or bipedal height of torso over feet
   double* torso_pos = data->xipos + 3*torso_body_id_;
   bool is_biped = current_mode_ == kModeBiped;
   double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
   if (current_mode_ == kModeScramble) {
    // disable height term in Scramble
     residual[counter++] = 0;
   } else if (current_mode_ == kModeFlip) {
    // height target for Backflip
     double flip_time = data->time - mode_start_time_;
     residual[counter++] = torso_pos[2] - FlipHeight(flip_time);
   } else {
     residual[counter++] = (torso_pos[2] - avg_foot_pos[2]) - height_goal;
   }


  // ---------- Position ----------
   double* head = data->site_xpos + 3*head_site_id_;
  double target[3];
  if (current_mode_ == kModeWalk) {
    // follow prescribed Walk trajectory
    double mode_time = data->time - mode_start_time_;
    Walk(target, mode_time);
  } else {
    // go to the goal mocap body
    target[0] = goal_pos[0];
    target[1] = goal_pos[1];
    target[2] = goal_pos[2];
  }
  // If pose is selected, zero out Position residual (replaced by Pose above)
  if (pose_target_selected) {
    residual[counter++] = 0;
    residual[counter++] = 0;
    residual[counter++] = 0;
  } else {
    residual[counter++] = head[0] - target[0];
    residual[counter++] = head[1] - target[1];
    residual[counter++] =
        current_mode_ == kModeScramble ? 2 * (head[2] - target[2]) : 0;
  }

  // ---------- Gait ----------
   A1Gait gait = GetGait();
   double step[kNumFoot];
   FootStep(step, GetPhase(data->time), gait);
   for (A1Foot foot : kFootAll) {
     if (is_biped) {
      // ignore "hands" in biped mode
       bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
       bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
       bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);
       if (front_hand || back_hand) {
         residual[counter++] = 0;
         continue;
       }
     }
     double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};

     if (current_mode_ == kModeScramble) {
       double torso_to_goal[3];
       double* goal = data->mocap_pos + 3*goal_mocap_id_;
       mju_sub3(torso_to_goal, goal, torso_pos);
       mju_normalize3(torso_to_goal);
       mju_sub3(torso_to_goal, goal, foot_pos[foot]);
       torso_to_goal[2] = 0;
       mju_normalize3(torso_to_goal);
       mju_addToScl3(query, torso_to_goal, 0.15);
     }

     double ground_height = Ground(model, data, query);
     double height_target = ground_height + kFootRadius + step[foot];
     double height_difference = foot_pos[foot][2] - height_target;
     if (current_mode_ == kModeScramble) {
      // in Scramble, foot higher than target is not penalized
       height_difference = mju_min(0, height_difference);
     }
     residual[counter++] = step[foot] ? height_difference : 0;
   }


   // ---------- Balance ----------
   double* comvel = SensorByName(model, data, "torso_subtreelinvel");
   double capture_point[3];
   double fall_time = mju_sqrt(2*height_goal / 9.81);
   mju_addScl3(capture_point, compos, comvel, fall_time);
   residual[counter++] = capture_point[0] - avg_foot_pos[0];
   residual[counter++] = capture_point[1] - avg_foot_pos[1];


   // ---------- Effort ----------
   mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
   counter += model->nu;


  // ---------- Posture ----------
   double* home = KeyQPosByName(model, data, "home");
   mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
   if (current_mode_ == kModeFlip) {
     double flip_time = data->time - mode_start_time_;
     if (flip_time < crouch_time_) {
       double* crouch = KeyQPosByName(model, data, "crouch");
       mju_sub(residual + counter, data->qpos + 7, crouch + 7, model->nu);
     } else if (flip_time >= crouch_time_ &&
                flip_time < jump_time_ + flight_time_) {
      // free legs during flight phase
       mju_zero(residual + counter, model->nu);
     }
   }
   for (A1Foot foot : kFootAll) {
     for (int joint = 0; joint < 3; joint++) {
       residual[counter + 3*foot + joint] *= kJointPostureGain[joint];
     }
   }
   if (current_mode_ == kModeBiped) {
    // loosen the "hands" in Biped mode
     bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
     double arm_posture = parameters_[arm_posture_param_id_];
     if (handstand) {
       residual[counter + 6] *= arm_posture;
       residual[counter + 7] *= arm_posture;
       residual[counter + 8] *= arm_posture;
       residual[counter + 9] *= arm_posture;
       residual[counter + 10] *= arm_posture;
       residual[counter + 11] *= arm_posture;
     } else {
       residual[counter + 0] *= arm_posture;
       residual[counter + 1] *= arm_posture;
       residual[counter + 2] *= arm_posture;
       residual[counter + 3] *= arm_posture;
       residual[counter + 4] *= arm_posture;
       residual[counter + 5] *= arm_posture;
     }
   }
   counter += model->nu;


  // ---------- Yaw ----------
   double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
   if (current_mode_ == kModeBiped) {
    int handstand =
        ReinterpretAsInt(parameters_[biped_type_param_id_]) ? 1 : -1;
     torso_heading[0] = handstand * torso_xmat[2];
     torso_heading[1] = handstand * torso_xmat[5];
   }
   mju_normalize(torso_heading, 2);
   double heading_goal = parameters_[ParameterIndex(model, "Heading")];
   residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
   residual[counter++] = torso_heading[1] - mju_sin(heading_goal);


  // ---------- Angular momentum ----------
   mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));
   counter +=3;


  // ---------- FootCost (placeholder to match user sensor dims) ----------
  // The GO2 model defines a trailing user sensor "FootCost" with dim=4.
  // QuadrupedPose does not compute it explicitly; append zeros to keep
  // total residual length equal to the sum of user sensor dimensions.
  mju_zero(residual + counter, 4);
  counter += 4;

  // ---------- NormClear (placeholder to match user sensor dims) ----------
  // If the GO2 model defines "NormClear" (dim=6), append zeros here.
  {
    int nc_id = CostTermByName(model, "NormClear");
    if (nc_id >= 0) {
      mju_zero(residual + counter, 6);
      counter += 6;
    }
  }


  // sensor dim sanity check
   CheckSensorDim(model, counter);
 }

 void QuadrupedPose::TransitionLocked(mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
   if (data->time < residual_.last_transition_time_ ||
       residual_.last_transition_time_ == -1) {
    if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
      mode = ResidualFn::kModeQuadruped;  // mode stateful, switch to Quadruped
     }
     residual_.last_transition_time_ = residual_.phase_start_time_ =
         residual_.phase_start_ = data->time;
   }

  // ---------- prevent forbidden mode transitions ----------
  // switching mode, not from quadruped
  if (mode != residual_.current_mode_ &&
      residual_.current_mode_ != ResidualFn::kModeQuadruped) {
    // switch into stateful mode only allowed from Quadruped
    if (mode == ResidualFn::kModeWalk || mode == ResidualFn::kModeFlip) {
      mode = ResidualFn::kModeQuadruped;
    }
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
    weight[residual_.height_cost_id_] = ResidualFn::kGaitParam[gait][5];
   }


  // ---------- Walk ----------
  double* goal_pos = data->mocap_pos + 3*residual_.goal_mocap_id_;
  if (mode == ResidualFn::kModeWalk) {
    double angvel = parameters[ParameterIndex(model, "Walk turn")];
    double speed = parameters[ParameterIndex(model, "Walk speed")];

    // current torso direction
    double* torso_xmat = data->xmat + 9*residual_.torso_body_id_;
    double forward[2] = {torso_xmat[0], torso_xmat[3]};
    mju_normalize(forward, 2);
    double leftward[2] = {-forward[1], forward[0]};

    // switching into Walk or parameters changed, reset task state
    if (mode != residual_.current_mode_ || residual_.angvel_ != angvel ||
        residual_.speed_ != speed) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save current speed and angvel
      residual_.speed_ = speed;
      residual_.angvel_ = angvel;

      // compute and save rotation axis / walk origin
      double axis[2] = {data->xpos[3*residual_.torso_body_id_],
                        data->xpos[3*residual_.torso_body_id_+1]};
      if (mju_abs(angvel) > ResidualFn::kMinAngvel) {
        // don't allow turning with very small angvel
        double d = speed / angvel;
        axis[0] += d * leftward[0];
        axis[1] += d * leftward[1];
      }
      residual_.position_[0] = axis[0];
      residual_.position_[1] = axis[1];

      // save vector from axis to initial goal position
      residual_.heading_[0] = goal_pos[0] - axis[0];
      residual_.heading_[1] = goal_pos[1] - axis[1];
    }

    // move goal
    double time = data->time - residual_.mode_start_time_;
    residual_.Walk(goal_pos, time);
  }


  // ---------- Flip ----------
  double* compos = SensorByName(model, data, "torso_subtreecom");
  if (mode == ResidualFn::kModeFlip) {
    // switching into Flip, reset task state
    if (mode != residual_.current_mode_) {
      // save time
      residual_.mode_start_time_ = data->time;

      // save body orientation, ground height
      mju_copy4(residual_.orientation_,
                data->xquat + 4 * residual_.torso_body_id_);
      residual_.ground_ = Ground(model, data, compos);

      // save parameters
      residual_.save_weight_ = weight;
      residual_.save_gait_switch_ = parameters[residual_.gait_switch_param_id_];

      // set parameters
      weight[CostTermByName(model, "Upright")] = 0.2;
      weight[CostTermByName(model, "Height")] = 5;
      weight[CostTermByName(model, "Position")] = 0;
      weight[CostTermByName(model, "Gait")] = 0;
      weight[CostTermByName(model, "Balance")] = 0;
      weight[CostTermByName(model, "Effort")] = 0.005;
      weight[CostTermByName(model, "Posture")] = 0.1;
      parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(0);
  }

    // time from start of Flip
    double flip_time = data->time - residual_.mode_start_time_;

    if (flip_time >=
        residual_.jump_time_ + residual_.flight_time_ + residual_.land_time_) {
      // Flip ended, back to Quadruped, restore values
      mode = ResidualFn::kModeQuadruped;
      weight = residual_.save_weight_;
      parameters[residual_.gait_switch_param_id_] = residual_.save_gait_switch_;
      goal_pos[0] = data->site_xpos[3*residual_.head_site_id_ + 0];
      goal_pos[1] = data->site_xpos[3*residual_.head_site_id_ + 1];
  }
  }

  // ---------- Visualize selected Pose target on mocap ----------
  // If a Pose is selected from the UI, set the 'goal' mocap to that keyframe
  // so it is rendered like in the hill task.
  {
    int pose_index = 0;
    if (residual_.pose_select_param_id_ >= 0) {
      pose_index = ReinterpretAsInt(parameters[residual_.pose_select_param_id_]);
    }
    if (pose_index > 0 && model->nkey >= pose_index + 1) {
      double* goal = data->mocap_pos + 3*residual_.goal_mocap_id_;
      mju_copy3(goal, model->key_mpos + 3 * pose_index);
      double* goal_quat = data->mocap_quat + 4*residual_.goal_mocap_id_;
      // For indices 1..4 (Position 1..4), we still set the mocap quaternion so it's visible,
      // but the Pose residual is disabled in Residual() for these indices.
      mju_copy4(goal_quat, model->key_mquat + 4 * pose_index);
    }
  }

  // save mode
   residual_.current_mode_ = static_cast<ResidualFn::A1Mode>(
       mjMIN(static_cast<int>(ResidualFn::kNumMode)-1, mjMAX(0, mode)));
   residual_.last_transition_time_ = data->time;
 }

 void QuadrupedPose::ResetLocked(const mjModel* model) {
   // ----------  task identifiers  ----------
   residual_.gait_param_id_ = ParameterIndex(model, "select_Gait");
   residual_.gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
   residual_.flip_dir_param_id_ = ParameterIndex(model, "select_Flip dir");
   residual_.biped_type_param_id_ = ParameterIndex(model, "select_Biped type");
   residual_.cadence_param_id_ = ParameterIndex(model, "Cadence");
   residual_.amplitude_param_id_ = ParameterIndex(model, "Amplitude");
   residual_.duty_param_id_ = ParameterIndex(model, "Duty ratio");
   residual_.arm_posture_param_id_ = ParameterIndex(model, "Arm posture");
   residual_.balance_cost_id_ = CostTermByName(model, "Balance");
   residual_.upright_cost_id_ = CostTermByName(model, "Upright");
   residual_.height_cost_id_ = CostTermByName(model, "Height");
  // Pose residual and selection
  residual_.pose_cost_id_ = CostTermByName(model, "Pose");
  residual_.pose_select_param_id_ = ParameterIndex(model, "select_Pose");

  // ---------- set initial weights (Quadruped Pose defaults) ----------
  // cost terms by name used on demand
  int position_cost_id = CostTermByName(model, "Position");
  int effort_cost_id = CostTermByName(model, "Effort");
  int posture_cost_id = CostTermByName(model, "Posture");
  int orientation_cost_id = CostTermByName(model, "Orientation");
  int angmom_cost_id = CostTermByName(model, "Angmom");
  if (residual_.pose_cost_id_ >= 0) weight[residual_.pose_cost_id_] = 0.05;  // Pose
  if (residual_.upright_cost_id_ >= 0) weight[residual_.upright_cost_id_] = 1.0;  // Upright
  if (residual_.height_cost_id_ >= 0) weight[residual_.height_cost_id_] = 1.0;    // Height
  if (position_cost_id >= 0) weight[position_cost_id] = 0.2;                      // Position
  if (residual_.balance_cost_id_ >= 0) weight[residual_.balance_cost_id_] = 0.2;  // Balance
  if (effort_cost_id >= 0) weight[effort_cost_id] = 0.08;                         // Effort
  if (posture_cost_id >= 0) weight[posture_cost_id] = 0.02;                       // Posture
  if (orientation_cost_id >= 0) weight[orientation_cost_id] = 0.0;                 // Orientation
  if (angmom_cost_id >= 0) weight[angmom_cost_id] = 0.0;                           // Angmom

  // ---------- set initial parameters (Quadruped Pose defaults) ----------
  // gait selection: Stand|Walk|Trot|Canter|Gallop  -> Trot = 2
  if (residual_.gait_param_id_ >= 0) {
    parameters[residual_.gait_param_id_] = ReinterpretAsDouble(ResidualFn::kGaitTrot);
    residual_.current_gait_ = ReinterpretAsDouble(ResidualFn::kGaitTrot);
  }
  // gait switch: Manual|Automatic -> Manual = 0
  if (residual_.gait_switch_param_id_ >= 0) {
    parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(0);
  }
  // cadence, amplitude, duty ratio
  if (residual_.cadence_param_id_ >= 0) parameters[residual_.cadence_param_id_] = 0.9;    // Trot cadence
  if (residual_.amplitude_param_id_ >= 0) parameters[residual_.amplitude_param_id_] = 0.03; // Trot amplitude
  if (residual_.duty_param_id_ >= 0) parameters[residual_.duty_param_id_] = 0.755;        // Trot duty ratio
  // walk speed/turn, heading, arm posture
  {
    int idx;
    idx = ParameterIndex(model, "Walk speed"); if (idx >= 0) parameters[idx] = 0.0;
    idx = ParameterIndex(model, "Walk turn");  if (idx >= 0) parameters[idx] = 0.0;
    idx = ParameterIndex(model, "Heading");    if (idx >= 0) parameters[idx] = 0.0;
  }
  if (residual_.arm_posture_param_id_ >= 0) parameters[residual_.arm_posture_param_id_] = 0.0;

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
    int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
    if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
    residual_.shoulder_body_id_[shoulder_index] = foot_id;
     shoulder_index++;
   }

   // ----------  derived kinematic quantities for Flip  ----------
   residual_.gravity_ = mju_norm3(model->opt.gravity);
  // velocity at takeoff
  residual_.jump_vel_ =
      mju_sqrt(2 * residual_.gravity_ *
               (ResidualFn::kMaxHeight - ResidualFn::kLeapHeight));
  // time in flight phase
   residual_.flight_time_ = 2 * residual_.jump_vel_ / residual_.gravity_;
  // acceleration during jump phase
  residual_.jump_acc_ =
      residual_.jump_vel_ * residual_.jump_vel_ /
      (2 * (ResidualFn::kLeapHeight - ResidualFn::kCrouchHeight));
  // time in crouch sub-phase of jump
  residual_.crouch_time_ =
      mju_sqrt(2 * (ResidualFn::kHeightQuadruped - ResidualFn::kCrouchHeight) /
                                     residual_.jump_acc_);
  // time in leap sub-phase of jump
   residual_.leap_time_ = residual_.jump_vel_ / residual_.jump_acc_;
  // jump total time
   residual_.jump_time_ = residual_.crouch_time_ + residual_.leap_time_;
  // velocity at beginning of crouch
   residual_.crouch_vel_ = -residual_.jump_acc_ * residual_.crouch_time_;
  // time of landing phase
  residual_.land_time_ =
      2 * (ResidualFn::kLeapHeight - ResidualFn::kHeightQuadruped) /
                          residual_.jump_vel_;
  // acceleration during landing
   residual_.land_acc_ = residual_.jump_vel_ / residual_.land_time_;
  // rotational velocity during flight phase (rotates 1.25 pi)
   residual_.flight_rot_vel_ = 1.25 * mjPI / residual_.flight_time_;
  // rotational velocity at start of leap (rotates 0.5 pi)
  residual_.jump_rot_vel_ =
      mjPI / residual_.leap_time_ - residual_.flight_rot_vel_;
  // rotational acceleration during leap (rotates 0.5 pi)
  residual_.jump_rot_acc_ =
      (residual_.flight_rot_vel_ - residual_.jump_rot_vel_) /
                             residual_.leap_time_;
  // rotational deceleration during land (rotates 0.25 pi)
  residual_.land_rot_acc_ =
      2 * (residual_.flight_rot_vel_ * residual_.land_time_ - mjPI / 4) /
                             (residual_.land_time_ * residual_.land_time_);
 }

  void QuadrupedPose::ModifyScene(const mjModel* model, const mjData* data,
                              mjvScene* scene) const {
  // flip target pose
  if (residual_.current_mode_ == ResidualFn::kModeFlip) {
    double flip_time = data->time - residual_.mode_start_time_;
    double* torso_pos = data->xpos + 3*residual_.torso_body_id_;
    double pos[3] = {torso_pos[0], torso_pos[1],
                     residual_.FlipHeight(flip_time)};
    double quat[4];
    residual_.FlipQuat(quat, flip_time);
    double mat[9];
    mju_quat2Mat(mat, quat);
    double size[3] = {0.25, 0.15, 0.05};
    float rgba[4] = {0, 1, 0, 0.5};
    AddGeom(scene, mjGEOM_BOX, size, pos, mat, rgba);
    return;
  }

  double* foot_pos[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * residual_.foot_geom_id_[foot];

  double flight_pos[ResidualFn::kNumFoot][3];
  double stance_pos[ResidualFn::kNumFoot][3];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    flight_pos[foot][0] = stance_pos[foot][0] = foot_pos[foot][0];
    flight_pos[foot][1] = stance_pos[foot][1] = foot_pos[foot][1];
  }

  double ground[ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    ground[foot] = Ground(model, data, foot_pos[foot]);
  }

  ResidualFn::A1Gait gait = residual_.GetGait();
  double step[ResidualFn::kNumFoot];
  residual_.FootStep(step, residual_.GetPhase(data->time), gait);

  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    stance_pos[foot][2] = ResidualFn::kFootRadius + ground[foot];
    if (residual_.current_mode_ == ResidualFn::kModeBiped) {
      bool handstand =
          ReinterpretAsInt(parameters[residual_.biped_type_param_id_]);
      bool front_hand = !handstand && (foot == ResidualFn::kFootFL ||
                                       foot == ResidualFn::kFootFR);
      bool back_hand = handstand && (foot == ResidualFn::kFootHL ||
                                     foot == ResidualFn::kFootHR);
      if (front_hand || back_hand) continue;
    }
    if (step[foot]) {
      flight_pos[foot][2] = ResidualFn::kFootRadius + step[foot] + ground[foot];
      AddConnector(scene, mjGEOM_CYLINDER, ResidualFn::kFootRadius,
                   stance_pos[foot], flight_pos[foot], kStepRgba);
    }
  }

  double polygon[2*ResidualFn::kNumFoot];
  for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
    polygon[2*foot] = foot_pos[foot][0];
    polygon[2*foot + 1] = foot_pos[foot][1];
  }
  int hull[ResidualFn::kNumFoot];
  int num_hull = Hull2D(hull, ResidualFn::kNumFoot, polygon);
  for (int i=0; i < num_hull; i++) {
    int j = (i + 1) % num_hull;
    AddConnector(scene, mjGEOM_CAPSULE, ResidualFn::kFootRadius/2,
                 stance_pos[hull[i]], stance_pos[hull[j]], kHullRgba);
  }

  bool is_biped = residual_.current_mode_ == ResidualFn::kModeBiped;
  double height_goal =
      is_biped ? ResidualFn::kHeightBiped : ResidualFn::kHeightQuadruped;
  double fall_time = mju_sqrt(2*height_goal / residual_.gravity_);
  double capture[3];
  double* compos = SensorByName(model, data, "torso_subtreecom");
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  mju_addScl3(capture, compos, comvel, fall_time);

  double com_ground = Ground(model, data, compos);

  double feet_pos[3];
  residual_.AverageFootPos(feet_pos, foot_pos);
  feet_pos[2] = com_ground;

  double foot_size[3] = {ResidualFn::kFootRadius, 0, 0};
  AddGeom(scene, mjGEOM_SPHERE, foot_size, feet_pos, /*mat=*/nullptr, kAvgRgba);

  capture[2] = com_ground;
  AddGeom(scene, mjGEOM_SPHERE, foot_size, capture, /*mat=*/nullptr, kCapRgba);

  double pcp2[2];
  NearestInHull(pcp2, capture, polygon, hull, num_hull);
  double pcp[3] = {pcp2[0], pcp2[1], com_ground};
  AddGeom(scene, mjGEOM_SPHERE, foot_size, pcp, /*mat=*/nullptr, kPcpRgba);
 }

// ===== Add missing helper definitions for QuadrupedPose::ResidualFn =====
void QuadrupedPose::ResidualFn::AverageFootPos(
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

double QuadrupedPose::ResidualFn::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// horizontal Walk trajectory
void QuadrupedPose::ResidualFn::Walk(double pos[2], double time) const {
  if (mju_abs(angvel_) < kMinAngvel) {
    // no rotation, go in straight line
    double forward[2] = {heading_[0], heading_[1]};
    mju_normalize(forward, 2);
    pos[0] = position_[0] + heading_[0] + time*speed_*forward[0];
    pos[1] = position_[1] + heading_[1] + time*speed_*forward[1];
  } else {
    // walk on a circle
    double angle = time * angvel_;
    double mat[4] = {mju_cos(angle), -mju_sin(angle),
                     mju_sin(angle),  mju_cos(angle)};
    mju_mulMatVec(pos, mat, heading_, 2, 2);
    pos[0] += position_[0];
    pos[1] += position_[1];
  }
}

QuadrupedPose::ResidualFn::A1Gait QuadrupedPose::ResidualFn::GetGait() const {
  if (current_mode_ == kModeBiped) return kGaitTrot;
  return static_cast<A1Gait>(ReinterpretAsInt(current_gait_));
}

double QuadrupedPose::ResidualFn::StepHeight(double time, double footphase,
                                             double duty_ratio) const {
  double angle = fmod(time + mjPI - footphase, 2*mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

void QuadrupedPose::ResidualFn::FootStep(double step[kNumFoot], double time,
                                         A1Gait gait) const {
  double amplitude = parameters_[amplitude_param_id_];
  double duty_ratio = parameters_[duty_param_id_];
  for (A1Foot foot : kFootAll) {
    double footphase = 2*mjPI*kGaitPhase[gait][foot];
    step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
  }
}

double QuadrupedPose::ResidualFn::FlipHeight(double time) const {
  if (time >= jump_time_ + flight_time_ + land_time_) {
    return kHeightQuadruped + ground_;
  }
  double h = 0;
  if (time < jump_time_) {
    h = kHeightQuadruped + time * crouch_vel_ + 0.5 * time * time * jump_acc_;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    h = kLeapHeight + jump_vel_*time - 0.5*9.81*time*time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    h = kLeapHeight - jump_vel_*time + 0.5*land_acc_*time*time;
  }
  return h + ground_;
}

void QuadrupedPose::ResidualFn::FlipQuat(double quat[4], double time) const {
  double angle = 0;
  if (time >= jump_time_ + flight_time_ + land_time_) {
    angle = 2*mjPI;
  } else if (time >= crouch_time_ && time < jump_time_) {
    time -= crouch_time_;
    angle = 0.5 * jump_rot_acc_ * time * time + jump_rot_vel_ * time;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    angle = mjPI/2 + flight_rot_vel_ * time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    angle = 1.75*mjPI + flight_rot_vel_*time - 0.5*land_rot_acc_ * time * time;
  }
  int flip_dir = ReinterpretAsInt(parameters_[flip_dir_param_id_]);
  double axis[3] = {0, flip_dir ? 1.0 : -1.0, 0};
  mju_axisAngle2Quat(quat, axis, angle);
  mju_mulQuat(quat, orientation_, quat);
}

std::string MjTwin::XmlPath() const {
  return GetModelPath("quadruped/task_mjTwin.xml");
}
std::string MjTwin::Name() const { return "mjTwin"; }

void MjTwin::ResetLocked(const mjModel* model) {
  // Perform the same identifier setup as QuadrupedFlat
  QuadrupedFlat::ResetLocked(model);

  // Mirror QuadrupedPose defaults for weights and parameters
  // cost terms by name used on demand
  int position_cost_id = CostTermByName(model, "Position");
  int effort_cost_id = CostTermByName(model, "Effort");
  int posture_cost_id = CostTermByName(model, "Posture");
  int orientation_cost_id = CostTermByName(model, "Orientation");
  int angmom_cost_id = CostTermByName(model, "Angmom");
  int upright_cost_id = CostTermByName(model, "Upright");
  int height_cost_id = CostTermByName(model, "Height");
  int balance_cost_id = CostTermByName(model, "Balance");
  int gait_cost_id = CostTermByName(model, "Gait");
  int footcost_cost_id = CostTermByName(model, "FootCost");

  // Note: MjTwin has no Pose residual; align other weights
  if (upright_cost_id >= 0) weight[upright_cost_id] = 0.195;            // Upright
  if (height_cost_id >= 0) weight[height_cost_id] = 0.0;                // Height
  if (position_cost_id >= 0) weight[position_cost_id] = 0.33;           // Position
  if (gait_cost_id >= 0) weight[gait_cost_id] = 1.0;                    // Gait
  if (balance_cost_id >= 0) weight[balance_cost_id] = 0.165;            // Balance
  if (effort_cost_id >= 0) weight[effort_cost_id] = 0.08;               // Effort
  if (posture_cost_id >= 0) weight[posture_cost_id] = 0.0605;           // Posture
  if (footcost_cost_id >= 0) weight[footcost_cost_id] = 0.065;          // FootCost
  if (orientation_cost_id >= 0) weight[orientation_cost_id] = 0.0;      // Orientation
  if (angmom_cost_id >= 0) weight[angmom_cost_id] = 0.0;                 // Angmom

  // Parameters: match QuadrupedPose defaults
  int gait_id = ParameterIndex(model, "select_Gait");
  int gait_switch_id = ParameterIndex(model, "select_Gait switch");
  int cadence_id = ParameterIndex(model, "Cadence");
  int amplitude_id = ParameterIndex(model, "Amplitude");
  int duty_id = ParameterIndex(model, "Duty ratio");
  int arm_posture_id = ParameterIndex(model, "Arm posture");

  if (gait_id >= 0) {
    // Trot index is 2: Stand|Walk|Trot|Canter|Gallop
    parameters[gait_id] = ReinterpretAsDouble(2);
    // current_gait_ is internal to ResidualFn; Transition will sync it on first call
  }
  if (gait_switch_id >= 0) {
    parameters[gait_switch_id] = ReinterpretAsDouble(0);  // Manual
  }
  if (cadence_id >= 0) parameters[cadence_id] = 0.9;      // Trot cadence
  if (amplitude_id >= 0) parameters[amplitude_id] = 0.03; // Trot amplitude
  if (duty_id >= 0) parameters[duty_id] = 0.755;          // Trot duty ratio

  {
    int idx;
    idx = ParameterIndex(model, "Walk speed"); if (idx >= 0) parameters[idx] = 0.0;
    idx = ParameterIndex(model, "Walk turn");  if (idx >= 0) parameters[idx] = 0.0;
    idx = ParameterIndex(model, "Heading");    if (idx >= 0) parameters[idx] = 0.0;
  }
  if (arm_posture_id >= 0) parameters[arm_posture_id] = 0.0;

  // If NormClear is present, enable it for MjTwin by setting a small default weight
  {
    int nc_id = CostTermByName(model, "NormClear");
    if (nc_id >= 0) {
      weight[nc_id] = 0.2;  // visible by default; tweakable
    }
  }

  // Cache terrain geom id for later world transforms
  cached_terrain_geom_id_ = mj_name2id(model, mjOBJ_GEOM, "terrain");

  // Cache head site and knee bodies for visualization
  head_site_id_vis_ = mj_name2id(model, mjOBJ_SITE, "head");
  knee_body_id_[0] = mj_name2id(model, mjOBJ_BODY, "FL_calf");
  knee_body_id_[1] = mj_name2id(model, mjOBJ_BODY, "FR_calf");
  knee_body_id_[2] = mj_name2id(model, mjOBJ_BODY, "HL_calf");
  knee_body_id_[3] = mj_name2id(model, mjOBJ_BODY, "HR_calf");


  
  // -------- Terrain hfield vertex normals (LOCAL hfield frame) --------
  terrain_normals_ = {};
  // Look up terrain hfield by name (matches QuadrupedFlat::ResetLocked)
  int hid = mj_name2id(model, mjOBJ_HFIELD, "hf133");
  if (hid >= 0) {
    int nrow = model->hfield_nrow[hid];
    int ncol = model->hfield_ncol[hid];
    int adr  = model->hfield_adr[hid];
    if (nrow > 0 && ncol > 0) {
      const double* hsize = model->hfield_size + 4 * hid;  // [sx, sy, sz, ...]
      double sx = hsize[0], sy = hsize[1], sz = hsize[2];
      // MuJoCo mapping: indices 0..ncol-1 span 2*sx in X; 0..nrow-1 span 2*sy in Y.
      // Compute grid spacing; derivatives are in the local geom frame (z-up).
      double dx = (ncol > 1) ? (2.0 * sx) / (ncol - 1) : (2.0 * sx);
      double dy = (nrow > 1) ? (2.0 * sy) / (nrow - 1) : (2.0 * sy);

      terrain_normals_.width = ncol;
      terrain_normals_.height = nrow;
      terrain_normals_.sx = sx;
      terrain_normals_.sy = sy;
      terrain_normals_.sz = sz;
      terrain_normals_.dx = dx;
      terrain_normals_.dy = dy;
      terrain_normals_.inv2dx = (dx > 0) ? (1.0 / (2.0 * dx)) : 0.0;
      terrain_normals_.inv2dy = (dy > 0) ? (1.0 / (2.0 * dy)) : 0.0;
      terrain_normals_.hfield_id = hid;
      terrain_normals_.data.resize(static_cast<size_t>(nrow) * ncol * 3);

      const float* H = model->hfield_data + adr;  // height in [0,1] typically
      auto heightAt = [&](int r, int c) -> double {
        r = mjMAX(0, mjMIN(r, nrow - 1));
        c = mjMAX(0, mjMIN(c, ncol - 1));
        return static_cast<double>(H[r * ncol + c]) * sz;
      };

      for (int r = 0; r < nrow; ++r) {
        for (int c = 0; c < ncol; ++c) {
          // One-sided at borders, central otherwise
          double hx;
          if (c == 0) {
            double h0 = heightAt(r, 0);
            double h1 = heightAt(r, mjMIN(1, ncol - 1));
            hx = (h1 - h0) / dx;
          } else if (c == ncol - 1) {
            double hn1 = heightAt(r, ncol - 1);
            double hn2 = heightAt(r, ncol - 2);
            hx = (hn1 - hn2) / dx;
          } else {
            double hm = heightAt(r, c - 1);
            double hp = heightAt(r, c + 1);
            hx = (hp - hm) * terrain_normals_.inv2dx;
          }

          double hy;
          if (r == 0) {
            double h0 = heightAt(0, c);
            double h1 = heightAt(mjMIN(1, nrow - 1), c);
            hy = (h1 - h0) / dy;
          } else if (r == nrow - 1) {
            double hn1 = heightAt(nrow - 1, c);
            double hn2 = heightAt(nrow - 2, c);
            hy = (hn1 - hn2) / dy;
          } else {
            double hm2 = heightAt(r - 1, c);
            double hp2 = heightAt(r + 1, c);
            hy = (hp2 - hm2) * terrain_normals_.inv2dy;
          }

          // unnormalized normal; z-up
          double nx = -hx;
          double ny = -hy;
          double nz = 1.0;
          double invlen = 1.0 / mju_sqrt(nx * nx + ny * ny + nz * nz + 1e-30);
          nx *= invlen; ny *= invlen; nz *= invlen;

          size_t idx = static_cast<size_t>(r) * ncol + static_cast<size_t>(c);
          float* out = &terrain_normals_.data[3 * idx];
          out[0] = static_cast<float>(nx);
          out[1] = static_cast<float>(ny);
          out[2] = static_cast<float>(nz);
        }
      }
    }
  }
}

void MjTwin::ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const {
  // draw base visuals from QuadrupedFlat
  QuadrupedFlat::ModifyScene(model, data, scene);

  // require a terrain geom and normals
  int terrain_gid = (cached_terrain_geom_id_ >= 0)
                        ? cached_terrain_geom_id_
                        : mj_name2id(model, mjOBJ_GEOM, "terrain");
  int hid = terrain_normals_.hfield_id;
  // Lazily initialize normals if missing
  if (hid < 0 || terrain_normals_.data.empty()) {
    int lazy_hid = mj_name2id(model, mjOBJ_HFIELD, "hf133");
    if (lazy_hid >= 0) {
      int nrow = model->hfield_nrow[lazy_hid];
      int ncol = model->hfield_ncol[lazy_hid];
      int adr  = model->hfield_adr[lazy_hid];
      if (nrow > 0 && ncol > 0) {
        const double* hsize = model->hfield_size + 4 * lazy_hid;
        double sx = hsize[0], sy = hsize[1], sz = hsize[2];
        double dx = (ncol > 1) ? (2.0 * sx) / (ncol - 1) : (2.0 * sx);
        double dy = (nrow > 1) ? (2.0 * sy) / (nrow - 1) : (2.0 * sy);

        // const_cast is safe here: we're filling a cache inside a const method
        auto* self = const_cast<MjTwin*>(this);
        self->terrain_normals_.width = ncol;
        self->terrain_normals_.height = nrow;
        self->terrain_normals_.sx = sx;
        self->terrain_normals_.sy = sy;
        self->terrain_normals_.sz = sz;
        self->terrain_normals_.dx = dx;
        self->terrain_normals_.dy = dy;
        self->terrain_normals_.inv2dx = (dx > 0) ? (1.0 / (2.0 * dx)) : 0.0;
        self->terrain_normals_.inv2dy = (dy > 0) ? (1.0 / (2.0 * dy)) : 0.0;
        self->terrain_normals_.hfield_id = lazy_hid;
        self->terrain_normals_.data.resize(static_cast<size_t>(nrow) * ncol * 3);

        const float* H = model->hfield_data + adr;
        auto heightAt = [&](int r, int c) -> double {
          r = mjMAX(0, mjMIN(r, nrow - 1));
          c = mjMAX(0, mjMIN(c, ncol - 1));
          return static_cast<double>(H[r * ncol + c]) * sz;
        };
        for (int r = 0; r < nrow; ++r) {
          for (int c = 0; c < ncol; ++c) {
            double hx;
            if (c == 0) {
              double h0 = heightAt(r, 0);
              double h1 = heightAt(r, mjMIN(1, ncol - 1));
              hx = (h1 - h0) / dx;
            } else if (c == ncol - 1) {
              double hn1 = heightAt(r, ncol - 1);
              double hn2 = heightAt(r, ncol - 2);
              hx = (hn1 - hn2) / dx;
            } else {
              double hm = heightAt(r, c - 1);
              double hp = heightAt(r, c + 1);
              hx = (hp - hm) * self->terrain_normals_.inv2dx;
            }

            double hy;
            if (r == 0) {
              double h0 = heightAt(0, c);
              double h1 = heightAt(mjMIN(1, nrow - 1), c);
              hy = (h1 - h0) / dy;
            } else if (r == nrow - 1) {
              double hn1 = heightAt(nrow - 1, c);
              double hn2 = heightAt(nrow - 2, c);
              hy = (hn1 - hn2) / dy;
            } else {
              double hm2 = heightAt(r - 1, c);
              double hp2 = heightAt(r + 1, c);
              hy = (hp2 - hm2) * self->terrain_normals_.inv2dy;
            }

            double nx = -hx, ny = -hy, nz = 1.0;
            double invlen = 1.0 / mju_sqrt(nx * nx + ny * ny + nz * nz + 1e-30);
            size_t idx = static_cast<size_t>(r) * ncol + static_cast<size_t>(c);
            self->terrain_normals_.data[3 * idx + 0] = static_cast<float>(nx * invlen);
            self->terrain_normals_.data[3 * idx + 1] = static_cast<float>(ny * invlen);
            self->terrain_normals_.data[3 * idx + 2] = static_cast<float>(nz * invlen);
          }
        }
      }
    }
    hid = terrain_normals_.hfield_id;
  }
  if (terrain_gid < 0 || hid < 0 || terrain_normals_.data.empty()) return;

  // terrain geom pose
  const double* gpos = data->geom_xpos + 3 * terrain_gid;
  const double* gmat = data->geom_xmat + 9 * terrain_gid;  // row-major 3x3

  // visualize 16 vertex normals in a central 4x4 grid
  int W = terrain_normals_.width;
  int H = terrain_normals_.height;
  if (W < 4 || H < 4) return;
  int col0 = (W - 4) / 2;
  int row0 = (H - 4) / 2;

  // geometry size parameters (make arrows evident)
  const float rgba[4] = {1.0f, 0.1f, 0.1f, 1.0f};
  double arrow_radius = 0.02;   // thicker capsule for visibility
  double arrow_len = 0.30;      // longer arrow for visibility
  double base_radius = 0.015;   // small sphere at base

  // iterate 4x4 central vertices
  for (int di = 0; di < 4; ++di) {
    for (int dj = 0; dj < 4; ++dj) {
      int col = col0 + dj;
      int row = row0 + di;
      const float* nlocf = TerrainNormalAt(col, row);
      if (!nlocf) continue;
      double nloc[3] = {nlocf[0], nlocf[1], nlocf[2]};

      // local vertex position in geom frame
      double x_local = -terrain_normals_.sx + col * terrain_normals_.dx;
      double y_local = -terrain_normals_.sy + row * terrain_normals_.dy;

      // sample height at the vertex (same as used for normals)
      int adr = model->hfield_adr[hid];
      int ncol = model->hfield_ncol[hid];
      const float* Hdata = model->hfield_data + adr;
      double z_local = static_cast<double>(Hdata[row * ncol + col]) * terrain_normals_.sz;

      // rotate position+normal to world
      double p_local[3] = {x_local, y_local, z_local};
      double p_world[3];
      double n_world[3];
      mju_mulMatVec(p_world, gmat, p_local, 3, 3);
      mju_mulMatVec(n_world, gmat, nloc, 3, 3);
      // translate to world position
      p_world[0] += gpos[0];
      p_world[1] += gpos[1];
      p_world[2] += gpos[2];

      // end point of the arrow
      double tip_world[3] = {p_world[0] + arrow_len * n_world[0],
                             p_world[1] + arrow_len * n_world[1],
                             p_world[2] + arrow_len * n_world[2]};

      // draw base sphere at the surface vertex
      double base_size[3] = {base_radius, 0, 0};
      AddGeom(scene, mjGEOM_SPHERE, base_size, p_world, /*mat=*/nullptr, rgba);

      // draw arrow as a capsule along the normal from vertex to tip
      AddConnector(scene, mjGEOM_CAPSULE, arrow_radius, p_world, tip_world, rgba);
    }
  }

  // ---- Visualize interpolated normals inside the same region ----
  // Bounds in local coordinates covering the 4x4 vertex block
  double x_min_local = -terrain_normals_.sx + col0 * terrain_normals_.dx;
  double x_max_local = -terrain_normals_.sx + (col0 + 3) * terrain_normals_.dx;
  double y_min_local = -terrain_normals_.sy + row0 * terrain_normals_.dy;
  double y_max_local = -terrain_normals_.sy + (row0 + 3) * terrain_normals_.dy;

  // Bilinear height sampler in local frame
  int ncol = model->hfield_ncol[hid];
  int nrow = model->hfield_nrow[hid];
  int adr  = model->hfield_adr[hid];
  const float* Hdata = model->hfield_data + adr;
  auto HeightBilinearLocal = [&](double x_local, double y_local) {
    double sx = terrain_normals_.sx, sy = terrain_normals_.sy;
    double dx = terrain_normals_.dx, dy = terrain_normals_.dy;
    double u = (x_local + sx) / dx;  // col
    double v = (y_local + sy) / dy;  // row
    int x0 = (int)mju_floor(u), y0 = (int)mju_floor(v);
    int x1 = x0 + 1, y1 = y0 + 1;
    double tx = u - x0, ty = v - y0;
    x0 = mjMAX(0, mjMIN(x0, ncol - 1));
    x1 = mjMAX(0, mjMIN(x1, ncol - 1));
    y0 = mjMAX(0, mjMIN(y0, nrow - 1));
    y1 = mjMAX(0, mjMIN(y1, nrow - 1));
    double h00 = Hdata[y0 * ncol + x0];
    double h10 = Hdata[y0 * ncol + x1];
    double h01 = Hdata[y1 * ncol + x0];
    double h11 = Hdata[y1 * ncol + x1];
    double h0 = (1.0 - tx) * h00 + tx * h10;
    double h1 = (1.0 - tx) * h01 + tx * h11;
    return ((1.0 - ty) * h0 + ty * h1) * terrain_normals_.sz;
  };

  // Interpolated arrows styling (distinct color and size)
  const float rgba_i[4] = {0.1f, 1.0f, 1.0f, 1.0f};
  double arrow_radius_i = 0.012;
  double arrow_len_i = 0.20;
  double base_radius_i = 0.010;

  // Sample a grid within the region to visualize bilinear normal interpolation
  int S = 7;  // samples per axis (including endpoints)
  for (int iy = 0; iy < S; ++iy) {
    double ty = (S == 1) ? 0.0 : (double)iy / (double)(S - 1);
    double y_loc = (1.0 - ty) * y_min_local + ty * y_max_local;
    for (int ix = 0; ix < S; ++ix) {
      double tx = (S == 1) ? 0.0 : (double)ix / (double)(S - 1);
      double x_loc = (1.0 - tx) * x_min_local + tx * x_max_local;

      double n_loc[3];
      if (!TerrainNormalBilinearLocal(x_loc, y_loc, n_loc)) continue;
      double z_loc = HeightBilinearLocal(x_loc, y_loc);

      double p_loc[3] = {x_loc, y_loc, z_loc};
      double p_w[3], n_w[3];
      mju_mulMatVec(p_w, gmat, p_loc, 3, 3);
      mju_mulMatVec(n_w, gmat, n_loc, 3, 3);
      p_w[0] += gpos[0];
      p_w[1] += gpos[1];
      p_w[2] += gpos[2];

      double tip_w[3] = {p_w[0] + arrow_len_i * n_w[0],
                         p_w[1] + arrow_len_i * n_w[1],
                         p_w[2] + arrow_len_i * n_w[2]};

      double base_size_i[3] = {base_radius_i, 0, 0};
      AddGeom(scene, mjGEOM_SPHERE, base_size_i, p_w, /*mat=*/nullptr, rgba_i);
      AddConnector(scene, mjGEOM_CAPSULE, arrow_radius_i, p_w, tip_w, rgba_i);
    }
  }

  // ---- Visualize 5 clearance sites: head + 4 knees ----
  const float rgba_head[4] = {1.0f, 0.2f, 0.2f, 1.0f};
  const float rgba_knee[4] = {1.0f, 0.2f, 0.2f, 1.0f};
  double site_r = 0.03;  // default knee radius; head is overridden below
  double sz3[3] = {site_r, 0, 0};

  // Head site position
  if (head_site_id_vis_ >= 0) {
    const double* p = data->site_xpos + 3 * head_site_id_vis_;
    // Use the exact head collision sphere radius if present: find sphere geom
    // attached to the trunk body and nearest the head site
    int trunk_bid = mj_name2id(model, mjOBJ_BODY, "trunk");
    double head_r = site_r;
    if (trunk_bid >= 0) {
      int best = -1;
      double bestd2 = 1e30;
      for (int gi = 0; gi < model->ngeom; ++gi) {
        if (model->geom_type[gi] != mjGEOM_SPHERE) continue;
        if (model->geom_bodyid[gi] != trunk_bid) continue;
        // prefer collision geoms
        if (model->geom_group[gi] != 3) continue;
        const double* gc = data->geom_xpos + 3 * gi;
        double dx = gc[0] - p[0];
        double dy = gc[1] - p[1];
        double dz = gc[2] - p[2];
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < bestd2) { bestd2 = d2; best = gi; }
      }
      if (best >= 0) head_r = model->geom_size[3 * best + 0];
    }
    double sz_head[3] = {head_r, 0, 0};
    AddGeom(scene, mjGEOM_SPHERE, sz_head, p, /*mat=*/nullptr, rgba_head);
  }
  // Knees: use body COM positions as proxies (calf bodies)
  for (int k = 0; k < 4; ++k) {
    int bid = knee_body_id_[k];
    if (bid < 0) continue;
    const double* p = data->xpos + 3 * bid;
    AddGeom(scene, mjGEOM_SPHERE, sz3, p, /*mat=*/nullptr, rgba_knee);
  }
}

bool MjTwin::TerrainNormalBilinearLocal(double x_local, double y_local, double n_local[3]) const {
  // Availability
  int W = terrain_normals_.width;
  int H = terrain_normals_.height;
  if (W <= 0 || H <= 0 || terrain_normals_.data.empty()) return false;

  // Map local x,y in [-sx, +sx], [-sy, +sy] to grid coordinates
  double sx = terrain_normals_.sx;
  double sy = terrain_normals_.sy;
  double dx = terrain_normals_.dx;
  double dy = terrain_normals_.dy;
  if (dx <= 0 || dy <= 0) return false;

  double u = (x_local + sx) / dx;  // col space
  double v = (y_local + sy) / dy;  // row space

  // Integer cell and weights
  int x0 = (int)mju_floor(u);
  int y0 = (int)mju_floor(v);
  int x1 = x0 + 1;
  int y1 = y0 + 1;
  double tx = u - x0;
  double ty = v - y0;

  // Clamp to valid bilinear neighborhood
  if (x0 < 0) { x0 = 0; x1 = 0; tx = 0.0; }
  if (y0 < 0) { y0 = 0; y1 = 0; ty = 0.0; }
  if (x1 >= W) { x1 = W - 1; x0 = W - 1; tx = 0.0; }
  if (y1 >= H) { y1 = H - 1; y0 = H - 1; ty = 0.0; }

  const float* n00 = TerrainNormalAt(x0, y0);
  const float* n10 = TerrainNormalAt(x1, y0);
  const float* n01 = TerrainNormalAt(x0, y1);
  const float* n11 = TerrainNormalAt(x1, y1);
  if (!n00 || !n10 || !n01 || !n11) return false;

  // Bilinear blend then renormalize (Phong-style normal interpolation)
  double nx0 = (1.0 - tx) * n00[0] + tx * n10[0];
  double ny0 = (1.0 - tx) * n00[1] + tx * n10[1];
  double nz0 = (1.0 - tx) * n00[2] + tx * n10[2];

  double nx1 = (1.0 - tx) * n01[0] + tx * n11[0];
  double ny1 = (1.0 - tx) * n01[1] + tx * n11[1];
  double nz1 = (1.0 - tx) * n01[2] + tx * n11[2];

  double nx = (1.0 - ty) * nx0 + ty * nx1;
  double ny = (1.0 - ty) * ny0 + ty * ny1;
  double nz = (1.0 - ty) * nz0 + ty * nz1;

  double invlen = 1.0 / mju_sqrt(nx * nx + ny * ny + nz * nz + 1e-30);
  n_local[0] = nx * invlen;
  n_local[1] = ny * invlen;
  n_local[2] = nz * invlen;
  return true;
}

bool MjTwin::TerrainNormalBilinearWorld(const mjModel* model, const mjData* data,
                                        double x_world, double y_world,
                                        double n_world[3]) const {
  // Find terrain geom and its pose
  int terrain_gid = cached_terrain_geom_id_ >= 0 ? cached_terrain_geom_id_
                                                 : mj_name2id(model, mjOBJ_GEOM, "terrain");
  if (terrain_gid < 0) return false;
  const double* gpos = data->geom_xpos + 3 * terrain_gid;
  const double* gmat = data->geom_xmat + 9 * terrain_gid;  // row-major

  // Transform world XY into local geom frame XY (ignore Z for projection)
  double p_world[3] = {x_world - gpos[0], y_world - gpos[1], 0.0};
  // Inverse rotation: n_local = R^T * p_world; since gmat is row-major, R^T is its transpose
  double p_local_xy[3];
  // Multiply by transpose: R^T * [x;y;0]
  p_local_xy[0] = gmat[0] * p_world[0] + gmat[3] * p_world[1] + gmat[6] * p_world[2];
  p_local_xy[1] = gmat[1] * p_world[0] + gmat[4] * p_world[1] + gmat[7] * p_world[2];
  p_local_xy[2] = 0.0;

  double n_local[3];
  if (!TerrainNormalBilinearLocal(p_local_xy[0], p_local_xy[1], n_local)) return false;

  // Rotate local normal to world: n_world = R * n_local
  mju_mulMatVec(n_world, gmat, n_local, 3, 3);
  // ensure unit length after numeric drift
  double inv = 1.0 / mju_sqrt(mju_dot(n_world, n_world, 3) + 1e-30);
  n_world[0] *= inv; n_world[1] *= inv; n_world[2] *= inv;
  return true;
}

bool MjTwin::TerrainHeightBilinearLocal(const mjModel* model,
                                        double x_local, double y_local,
                                        double& z_local) const {
  int W = terrain_normals_.width;
  int H = terrain_normals_.height;
  int hid = terrain_normals_.hfield_id;
  if (W <= 0 || H <= 0 || hid < 0) return false;
  int adr = model->hfield_adr[hid];
  const float* Hdata = model->hfield_data + adr;

  double u = (x_local + terrain_normals_.sx) / terrain_normals_.dx;
  double v = (y_local + terrain_normals_.sy) / terrain_normals_.dy;
  int x0 = (int)mju_floor(u), y0 = (int)mju_floor(v);
  int x1 = x0 + 1, y1 = y0 + 1;
  double tx = u - x0, ty = v - y0;
  x0 = mjMAX(0, mjMIN(x0, W - 1));
  x1 = mjMAX(0, mjMIN(x1, W - 1));
  y0 = mjMAX(0, mjMIN(y0, H - 1));
  y1 = mjMAX(0, mjMIN(y1, H - 1));

  double h00 = Hdata[y0 * W + x0];
  double h10 = Hdata[y0 * W + x1];
  double h01 = Hdata[y1 * W + x0];
  double h11 = Hdata[y1 * W + x1];
  double h0 = (1.0 - tx) * h00 + tx * h10;
  double h1 = (1.0 - tx) * h01 + tx * h11;
  double h = (1.0 - ty) * h0 + ty * h1;

  z_local = terrain_normals_.sz * h;
  return true;
}

bool MjTwin::TerrainSurfaceAndNormalWorld(const mjModel* model, const mjData* data,
                                          double x_world, double y_world,
                                          double s_world[3], double n_world[3]) const {
  int terrain_gid = cached_terrain_geom_id_ >= 0 ? cached_terrain_geom_id_
                                                 : mj_name2id(model, mjOBJ_GEOM, "terrain");
  if (terrain_gid < 0) return false;

  const double* R = data->geom_xmat + 9 * terrain_gid;
  const double* c = data->geom_xpos + 3 * terrain_gid;

  double pw[3] = {x_world - c[0], y_world - c[1], 0.0};
  double pl[3];
  // local = R^T * pw
  mju_mulMatTVec(pl, R, pw, 3, 3);

  double nL[3];
  if (!TerrainNormalBilinearLocal(pl[0], pl[1], nL)) return false;
  double zL;
  if (!TerrainHeightBilinearLocal(model, pl[0], pl[1], zL)) return false;

  double sL[3] = {pl[0], pl[1], zL};
  mju_mulMatVec(s_world, R, sL, 3, 3);
  s_world[0] += c[0];
  s_world[1] += c[1];
  s_world[2] += c[2];

  mju_mulMatVec(n_world, R, nL, 3, 3);
  double inv = 1.0 / mju_sqrt(mju_dot(n_world, n_world, 3) + 1e-30);
  n_world[0] *= inv;
  n_world[1] *= inv;
  n_world[2] *= inv;
  return true;
}

}  // namespace mjpc
