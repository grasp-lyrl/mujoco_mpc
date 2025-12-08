#include "mjpc/tasks/quadruped/quadruped.h"

#include <string>
#include <cmath>
#include <cstdio>

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc {

std::string Quadruped::XmlPath() const {
  return GetModelPath("quadruped/xmls/task_mjTwin.xml");
}

std::string Quadruped::Name() const { return "Quadruped Base"; }

// Return user-sensor id if present, otherwise -1 (no log spam).
static int OptionalCostTermByName(const mjModel* m, const std::string& name) {
  int id = mj_name2id(m, mjOBJ_SENSOR, name.c_str());
  if (id == -1 || m->sensor_type[id] != mjSENS_USER) {
    return -1;
  }
  return id;
}

void Quadruped::ResidualFn::Residual(const mjModel* model,
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
  if (current_mode_ == kModeBiped) {
    double biped_type = parameters_[biped_type_param_id_];
    int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
    residual[counter++] = torso_xmat[6] - handstand;
  } else {
    residual[counter++] = torso_xmat[8] - 1;
  }
  residual[counter++] = 0;
  residual[counter++] = 0;


  // ---------- Height ----------
  // quadrupedal or bipedal height of torso over feet
  double* torso_pos = data->xipos + 3*torso_body_id_;
  bool is_biped = current_mode_ == kModeBiped;
  double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
  if (current_mode_ == kModeScramble) {
    // disable height term in Scramble
    residual[counter++] = 0;
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

    // baseline ground under current foot position
    double ground_now = Ground(model, data, foot_pos[foot]);
    // ground at forward query (Scramble shifts it by 0.15 m toward goal)
    double ground_future = Ground(model, data, query);
    double height_target = ground_future + kFootRadius + step[foot];
    // If terrain changes > 2 cm between now and 15 cm forward, add extra 2 cm
    if (current_mode_ == kModeScramble) {
      if (mju_abs(ground_future - ground_now) > 0.02) {
        height_target += 0.02;
      }
    }
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

  // ---------- Terrain-normal clearance cost (optional) ----------
  if (clear_cost_id_ >= 0) {
    // Only compute for mjTwin; otherwise emit zeros to keep dims consistent
    auto twin = dynamic_cast<const MjTwin*>(task_);
    if (twin && terrain_geom_id_ >= 0) {
      constexpr double beta = 40.0;           // softer hinge for smoother gradients
      constexpr double margin = 0.02;         // require >= 2 cm clearance
      // knees: FL, FR, HL, HR (immaterial spheres centered at calf COM)
      for (int k = 0; k < 4; ++k) {
        int bid = knee_body_id_clear_[k];
        if (bid >= 0) {
          const double* pk = data->xpos + 3 * bid;
          bool ok = false;
          // Mesh-to-box distance via closest point on OBB surface
          double cp[3];
          ok = twin->BoxClosestSurfacePointForBody(model, data, bid, pk, cp);
          if (ok) {
            double d = mju_dist3(pk, cp);
            double u = std::log1p(mju_exp(beta * (margin - d))) / beta;
            residual[counter++] = u;
          } else {
            residual[counter++] = 0;
          }
        } else {
          residual[counter++] = 0;
        }
      }  // end for k

      // shoulders: FL, FR, HL, HR (shoulder bodies)
      for (int k = 0; k < 4; ++k) {
        int bid = shoulder_body_id_[k];
        if (bid >= 0) {
          const double* ps = data->xpos + 3 * bid;
          double cp[3];
          bool ok = twin->BoxClosestSurfacePointForBody(model, data, bid, ps, cp);
          if (ok) {
            double d = mju_dist3(ps, cp);
            double u = std::log1p(mju_exp(beta * (margin - d))) / beta;
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
        bool ok = false;
        double cp_cyl[3];
        ok = twin->BoxClosestSurfacePointForGeom(model, data, trunk_cyl_geom_id_clear_, pl, cp_cyl);
        if (ok) {
          double d = mju_dist3(pl, cp_cyl);
          double u = std::log1p(mju_exp(beta * (margin - d))) / beta;
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
        bool ok = false;
        double cp_sph[3];
        ok = twin->BoxClosestSurfacePointForGeom(model, data, trunk_sph_geom_id_clear_, pl, cp_sph);
        if (ok) {
          double d = mju_dist3(pl, cp_sph);
          double u = std::log1p(mju_exp(beta * (margin - d))) / beta;
          residual[counter++] = u;
        } else {
          residual[counter++] = 0;
        }
      } else {
        residual[counter++] = 0;
      }
    } else {
      // Non-mjTwin tasks or missing terrain: append zeros to match dims (4 knees + 4 shoulders + 2 trunk = 10)
      mju_zero(residual + counter, 10);
      counter += 10;
    }
  }

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

//  ============  transition  ============
void Quadruped::TransitionLocked(mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
  if (data->time < residual_.last_transition_time_ ||
      residual_.last_transition_time_ == -1) {
    if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
      if (dynamic_cast<const MjTwin*>(this)) {
        mode = ResidualFn::kModeScramble;
      } else {
        mode = ResidualFn::kModeQuadruped;  // mode stateful, switch to Quadruped
      }
    }
    residual_.last_transition_time_ = residual_.phase_start_time_ =
        residual_.phase_start_ = data->time;
  }

  // ---------- prevent forbidden mode transitions ----------
  // switching mode, not from quadruped
  if (mode != residual_.current_mode_ &&
      residual_.current_mode_ != ResidualFn::kModeQuadruped) {
    // switch into stateful mode only allowed from Quadruped
    if (mode == ResidualFn::kModeWalk) {
      mode = ResidualFn::kModeQuadruped;
    }
  }

  if (UseGaitParameters()) {
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
      auto cost_lookup = AllowMissingCostTerms() ? OptionalCostTermByName
                                                 : CostTermByName;
      ResidualFn::A1Gait gait = residual_.GetGait();
      parameters[residual_.duty_param_id_] = ResidualFn::kGaitParam[gait][0];
      parameters[residual_.cadence_param_id_] = ResidualFn::kGaitParam[gait][1];
      parameters[residual_.amplitude_param_id_] = ResidualFn::kGaitParam[gait][2];
      weight[residual_.balance_cost_id_] = ResidualFn::kGaitParam[gait][3];
      weight[residual_.upright_cost_id_] = ResidualFn::kGaitParam[gait][4];
      weight[residual_.height_cost_id_] = ResidualFn::kGaitParam[gait][5];
      int position_cost_id = cost_lookup(model, "Position");
      if (position_cost_id >= 0) {
        if (gait == ResidualFn::kGaitStand) {
          weight[position_cost_id] = 0.0;
        } else {
          weight[position_cost_id] = ResidualFn::kPositionWeightDefault;
        }
      }
    }
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

  // save mode
  residual_.current_mode_ = static_cast<ResidualFn::A1Mode>(
      mjMIN(static_cast<int>(ResidualFn::kNumMode)-1, mjMAX(0, mode)));
  residual_.last_transition_time_ = data->time;
}

void Quadruped::ResetLocked(const mjModel* model) {
  // ----------  task identifiers  ----------
  if (UseGaitParameters()) {
    residual_.gait_param_id_ = ParameterIndex(model, "select_Gait");
    residual_.gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
    residual_.biped_type_param_id_ = ParameterIndex(model, "select_Biped type");
    residual_.cadence_param_id_ = ParameterIndex(model, "Cadence");
    residual_.amplitude_param_id_ = ParameterIndex(model, "Amplitude");
    residual_.duty_param_id_ = ParameterIndex(model, "Duty ratio");
    residual_.arm_posture_param_id_ = ParameterIndex(model, "Arm posture");
  } else {
    residual_.gait_param_id_ = -1;
    residual_.gait_switch_param_id_ = -1;
    residual_.biped_type_param_id_ = -1;
    residual_.cadence_param_id_ = -1;
    residual_.amplitude_param_id_ = -1;
    residual_.duty_param_id_ = -1;
    residual_.arm_posture_param_id_ = -1;
  }
  auto cost_lookup = AllowMissingCostTerms()
                     ? OptionalCostTermByName
                     : CostTermByName;
  residual_.balance_cost_id_ = cost_lookup(model, "Balance");
  residual_.upright_cost_id_ = cost_lookup(model, "Upright");
  residual_.height_cost_id_ = cost_lookup(model, "Height");

  // clearance cost term id (optional; user sensor named "NormClear")
  residual_.clear_cost_id_ = AllowMissingCostTerms()
                                 ? OptionalCostTermByName(model, "NormClear")
                                 : CostTermByName(model, "NormClear");
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

  int shoulder_index = 0;
  for (const char* shouldername : {"FL_hip", "HL_hip", "FR_hip", "HR_hip"}) {
    int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
    if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
    residual_.shoulder_body_id_[shoulder_index] = foot_id;
    shoulder_index++;
  }

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

// return phase as a function of time
double Quadruped::ResidualFn::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// horizontal Walk trajectory
void Quadruped::ResidualFn::Walk(double pos[2], double time) const {
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
Quadruped::ResidualFn::A1Gait Quadruped::ResidualFn::GetGait() const {
  if (current_mode_ == kModeBiped)
    return kGaitTrot;
  return static_cast<A1Gait>(ReinterpretAsInt(current_gait_));
}

// return normalized target step height
double Quadruped::ResidualFn::StepHeight(double time, double footphase,
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
void Quadruped::ResidualFn::FootStep(double step[kNumFoot], double time,
                                         A1Gait gait) const {
  double amplitude = parameters_[amplitude_param_id_];
  double duty_ratio = parameters_[duty_param_id_];
  for (A1Foot foot : kFootAll) {
    double footphase = 2*mjPI*kGaitPhase[gait][foot];
    step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
  }
}

std::string MjTwin::XmlPath() const {
  return GetModelPath("quadruped/xmls/task_mjTwin.xml");
}
std::string MjTwin::Name() const { return "mjTwin"; }

void MjTwin::ResetLocked(const mjModel* model) {

  Quadruped::ResetLocked(model);
  terrain_ = Terrain(model);

  // Default to Scramble mode for mjTwin (Quadruped|Biped|Walk|Scramble => 3)
  mode = 3;

  auto cost_lookup = AllowMissingCostTerms() ? OptionalCostTermByName : CostTermByName;

  int position_cost_id    = cost_lookup(model, "Position");
  int effort_cost_id      = cost_lookup(model, "Effort");
  int posture_cost_id     = cost_lookup(model, "Posture");
  // int orientation_cost_id = cost_lookup(model, "Orientation");
  // int angmom_cost_id      = cost_lookup(model, "Angmom");
  // int upright_cost_id     = cost_lookup(model, "Upright");
  int height_cost_id      = cost_lookup(model, "Height");
  int balance_cost_id     = cost_lookup(model, "Balance");
  // int gait_cost_id        = cost_lookup(model, "Gait");
  int normclear_cost_id    = cost_lookup(model, "NormClear");

  // if (upright_cost_id >= 0)     weight[upright_cost_id] = 0.195;          
  if (height_cost_id >= 0)      weight[height_cost_id] = 0.0;              
  if (position_cost_id >= 0)    weight[position_cost_id] = ResidualFn::kPositionWeightDefault;           
  // if (gait_cost_id >= 0)        weight[gait_cost_id] = 1.0;                
  if (balance_cost_id >= 0)     weight[balance_cost_id] = 0.21;           
  if (effort_cost_id >= 0)      weight[effort_cost_id] = 0.08;             
  if (posture_cost_id >= 0)     weight[posture_cost_id] = 0.03;          
  if (normclear_cost_id >= 0)   weight[normclear_cost_id] = 5.0;           
  // if (orientation_cost_id >= 0) weight[orientation_cost_id] = 0.0;        
  // if (angmom_cost_id >= 0)      weight[angmom_cost_id] = 0.0;            

  if (UseGaitParameters()) {
    int gait_id        = ParameterIndex(model, "select_Gait");
    int gait_switch_id = ParameterIndex(model, "select_Gait switch");
    int cadence_id     = ParameterIndex(model, "Cadence");
    int amplitude_id   = ParameterIndex(model, "Amplitude");
    int duty_id        = ParameterIndex(model, "Duty ratio");
    int arm_posture_id = ParameterIndex(model, "Arm posture");

    if (gait_id >= 0)        parameters[gait_id]        = ReinterpretAsDouble(2); // trot
    if (gait_switch_id >= 0) parameters[gait_switch_id] = ReinterpretAsDouble(0); // Manual}
    if (cadence_id >= 0)     parameters[cadence_id]     = 1.0;            // Trot cadence
    if (amplitude_id >= 0)   parameters[amplitude_id]   = 0.03;           // Trot amplitude
    if (duty_id >= 0)        parameters[duty_id]        = 0.8;          // Trot duty ratio

    {
      int idx;
      idx = ParameterIndex(model, "Walk speed"); if (idx >= 0) parameters[idx] = 0.0;
      idx = ParameterIndex(model, "Walk turn");  if (idx >= 0) parameters[idx] = 0.0;
      idx = ParameterIndex(model, "Heading");    if (idx >= 0) parameters[idx] = 0.0;
    }
    if (arm_posture_id >= 0) parameters[arm_posture_id] = 0.0;
  }

  // Cache head site and knee bodies for visualization
  head_site_id_vis_ = mj_name2id(model, mjOBJ_SITE, "head");
  knee_body_id_[0] = mj_name2id(model, mjOBJ_BODY, "FL_calf");
  knee_body_id_[1] = mj_name2id(model, mjOBJ_BODY, "FR_calf");
  knee_body_id_[2] = mj_name2id(model, mjOBJ_BODY, "HL_calf");
  knee_body_id_[3] = mj_name2id(model, mjOBJ_BODY, "HR_calf");


  
  // -------- Terrain hfield cache (heights + vertex normals) --------
  terrain_.Initialize(model);

  // Build generic pairs for any mocap body named "box_<geomname>"
  generic_pairs_.clear();
  for (int gi = 0; gi < model->ngeom; ++gi) {
    const char* gname = model->names + model->name_geomadr[gi];
    if (!gname || !*gname) continue;
    char boxname[256];
    std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
    int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
    if (bid >= 0) {
      int mid = model->body_mocapid[bid];
      if (mid >= 0) {
        PairMapEntry e;
        e.geom_id = gi;
        e.mocap_id = mid;
        // derive half height from the mocap's first geom if we can find it later in Transition
        e.half_h = 0.02;
        generic_pairs_.push_back(e);
      }
    }
  }
}

// Update mocap boxes under feet with top faces tangent to local hfield
void MjTwin::TransitionLocked(mjModel* model, mjData* data) {
  // Clamp unsupported mode indices to Scramble.
  // Valid modes: 0:Quadruped, 1:Biped, 2:Walk, 3:Scramble
  if (mode >= 4) {
    mode = 3;
  }

  // Call base to keep existing behavior (gait, goals, etc.)
  Quadruped::TransitionLocked(model, data);

  // Lazy init: cache mocap ids and foot geoms on first call
  if (box_mocap_id_[0] < 0) {

    const char* box_names[4]       = {"box_FL", "box_FR", "box_HL", "box_HR"};
    const char* foot_geom_names[4] = {"FL", "FR", "HL", "HR"};
    
    for (int i = 0; i < 4; ++i) {
      int bid = mj_name2id(model, mjOBJ_XBODY, box_names[i]);
      if (bid >= 0) {
        box_mocap_id_[i] = model->body_mocapid[bid];
      }
      foot_geom_id_boxref_[i] = mj_name2id(model, mjOBJ_GEOM, foot_geom_names[i]);
    }
    // detect half-height from the first box geom if available
    int g0 = mj_name2id(model, mjOBJ_GEOM, "box_FL_geom");
    if (g0 >= 0 && model->geom_type[g0] == mjGEOM_BOX) {
      box_half_height_ = model->geom_size[3 * g0 + 2];
    }
  }

  // For each foot: sample surface and normal, place and orient box
  for (int i = 0; i < 4; ++i) {
    int mid = box_mocap_id_[i];
    int gid = foot_geom_id_boxref_[i];
    if (mid < 0 || gid < 0) continue;

    // foot world position
    const double* pf = data->geom_xpos + 3 * gid;

    // sample terrain surface and normal at foot XY
    double s_world[3], n_world[3];
    double z_height;
    if (!terrain_.GetHeightFromWorld(data, pf[0], pf[1], z_height)) continue;
    s_world[0] = pf[0];
    s_world[1] = pf[1];
    s_world[2] = z_height;
    terrain_.GetNormalFromWorld(data, pf[0], pf[1], n_world);

    // set box center: on surface minus half-height along normal
    data->mocap_pos[3 * mid + 0] = s_world[0] - box_half_height_ * n_world[0];
    data->mocap_pos[3 * mid + 1] = s_world[1] - box_half_height_ * n_world[1];
    data->mocap_pos[3 * mid + 2] = s_world[2] - box_half_height_ * n_world[2];

    // orient box so its local +Z aligns with normal; choose a stable tangent X
    // Build orthonormal basis (x, y, z) with z = n_world
    double z_axis[3] = {n_world[0], n_world[1], n_world[2]};
    // pick arbitrary up to avoid near-collinearity
    double a[3] = {0.0, 0.0, 1.0};
    if (mju_abs(z_axis[2]) > 0.9) a[0] = 1.0, a[1] = 0.0, a[2] = 0.0;
    double x[3];
    mju_cross(x, a, z_axis);  // x = a x z
    double nx = mju_norm3(x);
    if (nx < 1e-9) {
      // fallback to global X if degenerate
      x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
    } else {
      mju_scl3(x, x, 1.0 / nx);
    }
    double y[3];
    mju_cross(y, z_axis, x);  // y = z x x

    // rotation matrix R = [x y z] in columns; convert to quaternion
    double R[9] = {x[0], y[0], z_axis[0],
                   x[1], y[1], z_axis[1],
                   x[2], y[2], z_axis[2]};
    double q[4];
    mju_mat2Quat(q, R);
    data->mocap_quat[4 * mid + 0] = q[0];
    data->mocap_quat[4 * mid + 1] = q[1];
    data->mocap_quat[4 * mid + 2] = q[2];
    data->mocap_quat[4 * mid + 3] = q[3];
  }

  // Update generic pairs (other robot geoms to their designated boxes)
  for (auto& e : generic_pairs_) {
    if (e.geom_id < 0 || e.mocap_id < 0) continue;
    const double* pg = data->geom_xpos + 3 * e.geom_id;
    double s_world[3], n_world[3];
    double z_height;
    if (!terrain_.GetHeightFromWorld(data, pg[0], pg[1], z_height)) continue;
    s_world[0] = pg[0];
    s_world[1] = pg[1];
    s_world[2] = z_height;
    terrain_.GetNormalFromWorld(data, pg[0], pg[1], n_world);

    // lazily fetch half-height from the mocap's attached geom size if unknown
    if (e.half_h <= 0.0) {
      // try to locate a geom by name "box_<geomname>_geom"
      const char* gname = model->names + model->name_geomadr[e.geom_id];
      char gbox[256];
      std::snprintf(gbox, sizeof(gbox), "box_%s_geom", gname);
      int gbox_id = mj_name2id(model, mjOBJ_GEOM, gbox);
      if (gbox_id >= 0 && model->geom_type[gbox_id] == mjGEOM_BOX) {
        e.half_h = model->geom_size[3 * gbox_id + 2];
      } else {
        e.half_h = box_half_height_;
      }
    }

    data->mocap_pos[3 * e.mocap_id + 0] = s_world[0] - e.half_h * n_world[0];
    data->mocap_pos[3 * e.mocap_id + 1] = s_world[1] - e.half_h * n_world[1];
    data->mocap_pos[3 * e.mocap_id + 2] = s_world[2] - e.half_h * n_world[2];

    double z_axis[3] = {n_world[0], n_world[1], n_world[2]};
    double a[3] = {0.0, 0.0, 1.0};
    if (mju_abs(z_axis[2]) > 0.9) a[0] = 1.0, a[1] = 0.0, a[2] = 0.0;
    double x[3];
    mju_cross(x, a, z_axis);
    double nx = mju_norm3(x);
    if (nx < 1e-9) { x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; }
    else { mju_scl3(x, x, 1.0 / nx); }
    double y[3];
    mju_cross(y, z_axis, x);
    double R[9] = {x[0], y[0], z_axis[0], x[1], y[1], z_axis[1], x[2], y[2], z_axis[2]};
    double q[4];
    mju_mat2Quat(q, R);
    mju_copy(data->mocap_quat + 4 * e.mocap_id, q, 4);
  }
}

// Compute top surface point and normal for mocap box corresponding to geom_id.
// We use the mocap body's pose and its first attached box geom's half-height.
bool MjTwin::BoxTopSurfaceAndNormalForGeom(const mjModel* model, const mjData* data,
                                           int geom_id,
                                           double s_world[3], double n_world[3]) const {
  if (geom_id < 0) return false;
  const char* gname = model->names + model->name_geomadr[geom_id];
  if (!gname || !*gname) return false;
  char boxname[256];
  std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
  int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
  if (bid < 0) return false;
  int mid = model->body_mocapid[bid];
  if (mid < 0) return false;

  // Find the box geom for this mocap, named "box_<gname>_geom" if present
  double half_h = box_half_height_;
  char gboxname[256];
  std::snprintf(gboxname, sizeof(gboxname), "box_%s_geom", gname);
  int gbox_id = mj_name2id(model, mjOBJ_GEOM, gboxname);
  if (gbox_id >= 0 && model->geom_type[gbox_id] == mjGEOM_BOX) {
    half_h = model->geom_size[3 * gbox_id + 2];
  }

  // Mocap pose
  const double* p = data->mocap_pos + 3 * mid;
  const double* q = data->mocap_quat + 4 * mid;
  double R[9];
  mju_quat2Mat(R, q);

  // Box local +Z is its top normal in our convention
  n_world[0] = R[2];
  n_world[1] = R[5];
  n_world[2] = R[8];
  // Top surface point = center + half_h * n
  s_world[0] = p[0] + half_h * n_world[0];
  s_world[1] = p[1] + half_h * n_world[1];
  s_world[2] = p[2] + half_h * n_world[2];
  return true;
}

bool MjTwin::BoxTopSurfaceAndNormalForBody(const mjModel* model, const mjData* data,
                                            int body_id,
                                            double s_world[3], double n_world[3]) const {
  if (body_id < 0) return false;
  // Iterate geoms to find a named collision geom on this body, prefer group 3
  int best_gi = -1;
  for (int gi = 0; gi < model->ngeom; ++gi) {
    if (model->geom_bodyid[gi] != body_id) continue;
    const char* gname = model->names + model->name_geomadr[gi];
    if (!gname || !*gname) continue;
    // Expect a mapping box_<gname>
    char boxname[256];
    std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
    int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
    if (bid >= 0) { best_gi = gi; break; }
  }
  if (best_gi < 0) return false;
  return BoxTopSurfaceAndNormalForGeom(model, data, best_gi, s_world, n_world);
}

// Ray from box center towards p_world; find intersection with oriented box.
// Box defined by mocap pose (R, center) and half-sizes from box geom.
static bool RayOBBSurfaceIntersection(const double box_p[3], const double box_R[9],
                                      const double half[3],
                                      const double p_world[3],
                                      double s_world[3]) {
  // Transform point and ray to box local frame: ray origin at (0,0,0), direction to p_local
  // p_local = R^T * (p_world - box_p)
  double pw[3] = {p_world[0] - box_p[0], p_world[1] - box_p[1], p_world[2] - box_p[2]};
  double RT[9] = {box_R[0], box_R[3], box_R[6],
                  box_R[1], box_R[4], box_R[7],
                  box_R[2], box_R[5], box_R[8]};
  double p_local[3] = {
    RT[0]*pw[0] + RT[1]*pw[1] + RT[2]*pw[2],
    RT[3]*pw[0] + RT[4]*pw[1] + RT[5]*pw[2],
    RT[6]*pw[0] + RT[7]*pw[1] + RT[8]*pw[2]
  };
  // Direction from center to point
  double dir[3] = {p_local[0], p_local[1], p_local[2]};
  double len = mju_norm3(dir);
  if (len < 1e-12) return false;
  mju_scl3(dir, dir, 1.0 / len);
  // Compute t where ray hits each slab x=±half[0], y=±half[1], z=±half[2]
  double tmin = 0.0, tmax = 1e12;
  for (int i = 0; i < 3; ++i) {
    double d = dir[i];
    double h = half[i];
    if (mju_abs(d) < 1e-12) {
      // Parallel to slabs; must be within bounds to have intersection, but origin at center so ok
      continue;
    }
    double t1 = (-h) / d;
    double t2 = ( h) / d;
    if (t1 > t2) { double tmp = t1; t1 = t2; t2 = tmp; }
    if (t1 > tmin) tmin = t1;
    if (t2 < tmax) tmax = t2;
  }
  // First positive intersection along ray direction
  double t = tmax;
  if (t < 0) return false;
  double hit_local[3] = {t * dir[0], t * dir[1], t * dir[2]};
  // Back to world: s_world = box_p + R * hit_local
  s_world[0] = box_R[0]*hit_local[0] + box_R[1]*hit_local[1] + box_R[2]*hit_local[2] + box_p[0];
  s_world[1] = box_R[3]*hit_local[0] + box_R[4]*hit_local[1] + box_R[5]*hit_local[2] + box_p[1];
  s_world[2] = box_R[6]*hit_local[0] + box_R[7]*hit_local[1] + box_R[8]*hit_local[2] + box_p[2];
  return true;
}

// Project point to inside the OBB, then clamp to box, then bring to surface.
static bool ClosestPointOnOBBSurface(const double box_p[3], const double box_R[9],
                                     const double half[3],
                                     const double p_world[3],
                                     double s_world[3]) {
  // Local point pl = R^T (p - box_p)
  double pw[3] = {p_world[0] - box_p[0], p_world[1] - box_p[1], p_world[2] - box_p[2]};
  double RT[9] = {box_R[0], box_R[3], box_R[6],
                  box_R[1], box_R[4], box_R[7],
                  box_R[2], box_R[5], box_R[8]};
  double pl[3] = {
    RT[0]*pw[0] + RT[1]*pw[1] + RT[2]*pw[2],
    RT[3]*pw[0] + RT[4]*pw[1] + RT[5]*pw[2],
    RT[6]*pw[0] + RT[7]*pw[1] + RT[8]*pw[2]
  };

  // Clamp to box extents
  double qc[3];
  for (int i = 0; i < 3; ++i) {
    if (pl[i] >  half[i]) qc[i] =  half[i];
    else if (pl[i] < -half[i]) qc[i] = -half[i];
    else qc[i] = pl[i];
  }

  // If inside (|pl[i]| < half[i] for all i), push out along the largest penetration axis
  bool inside = (mju_abs(pl[0]) <= half[0] && mju_abs(pl[1]) <= half[1] && mju_abs(pl[2]) <= half[2]);
  if (inside) {
    // Choose axis with smallest margin to face (closest surface)
    double dx = half[0] - mju_abs(pl[0]);
    double dy = half[1] - mju_abs(pl[1]);
    double dz = half[2] - mju_abs(pl[2]);
    if (dx <= dy && dx <= dz) qc[0] = (pl[0] >= 0 ? half[0] : -half[0]);
    else if (dy <= dx && dy <= dz) qc[1] = (pl[1] >= 0 ? half[1] : -half[1]);
    else qc[2] = (pl[2] >= 0 ? half[2] : -half[2]);
  }

  // Back to world: s_world = box_p + R * qc
  s_world[0] = box_R[0]*qc[0] + box_R[1]*qc[1] + box_R[2]*qc[2] + box_p[0];
  s_world[1] = box_R[3]*qc[0] + box_R[4]*qc[1] + box_R[5]*qc[2] + box_p[1];
  s_world[2] = box_R[6]*qc[0] + box_R[7]*qc[1] + box_R[8]*qc[2] + box_p[2];
  return true;
}

bool MjTwin::BoxClosestSurfacePointForGeom(const mjModel* model, const mjData* data,
                                           int geom_id, const double p_world[3],
                                           double s_world[3]) const {
  if (geom_id < 0) return false;
  const char* gname = model->names + model->name_geomadr[geom_id];
  if (!gname || !*gname) return false;
  char boxname[256];
  std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
  int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
  if (bid < 0) return false;
  int mid = model->body_mocapid[bid];
  if (mid < 0) return false;

  const double* p = data->mocap_pos + 3 * mid;
  const double* q = data->mocap_quat + 4 * mid;
  double R[9];
  mju_quat2Mat(R, q);
  double half[3] = {box_half_height_, box_half_height_, box_half_height_};
  char gboxname[256];
  std::snprintf(gboxname, sizeof(gboxname), "box_%s_geom", gname);
  int gbox_id = mj_name2id(model, mjOBJ_GEOM, gboxname);
  if (gbox_id >= 0 && model->geom_type[gbox_id] == mjGEOM_BOX) {
    half[0] = model->geom_size[3 * gbox_id + 0];
    half[1] = model->geom_size[3 * gbox_id + 1];
    half[2] = model->geom_size[3 * gbox_id + 2];
  }
  return ClosestPointOnOBBSurface(p, R, half, p_world, s_world);
}

bool MjTwin::BoxClosestSurfacePointForBody(const mjModel* model, const mjData* data,
                                           int body_id, const double p_world[3],
                                           double s_world[3]) const {
  if (body_id < 0) return false;
  int best_gi = -1;
  for (int gi = 0; gi < model->ngeom; ++gi) {
    if (model->geom_bodyid[gi] != body_id) continue;
    const char* gname = model->names + model->name_geomadr[gi];
    if (!gname || !*gname) continue;
    char boxname[256];
    std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
    int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
    if (bid >= 0) { best_gi = gi; break; }
  }
  if (best_gi < 0) return false;
  return BoxClosestSurfacePointForGeom(model, data, best_gi, p_world, s_world);
}
bool MjTwin::BoxCenterRaySurfacePointForGeom(const mjModel* model, const mjData* data,
                                             int geom_id, const double p_world[3],
                                             double s_world[3]) const {
  if (geom_id < 0) return false;
  const char* gname = model->names + model->name_geomadr[geom_id];
  if (!gname || !*gname) return false;
  char boxname[256];
  std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
  int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
  if (bid < 0) return false;
  int mid = model->body_mocapid[bid];
  if (mid < 0) return false;

  // Mocap pose
  const double* p = data->mocap_pos + 3 * mid;
  const double* q = data->mocap_quat + 4 * mid;
  double R[9];
  mju_quat2Mat(R, q);

  // Box half-sizes from attached box geom if available
  double half[3] = {box_half_height_, box_half_height_, box_half_height_};
  char gboxname[256];
  std::snprintf(gboxname, sizeof(gboxname), "box_%s_geom", gname);
  int gbox_id = mj_name2id(model, mjOBJ_GEOM, gboxname);
  if (gbox_id >= 0 && model->geom_type[gbox_id] == mjGEOM_BOX) {
    half[0] = model->geom_size[3 * gbox_id + 0];
    half[1] = model->geom_size[3 * gbox_id + 1];
    half[2] = model->geom_size[3 * gbox_id + 2];
  }
  return RayOBBSurfaceIntersection(p, R, half, p_world, s_world);
}

bool MjTwin::BoxCenterRaySurfacePointForBody(const mjModel* model, const mjData* data,
                                             int body_id, const double p_world[3],
                                             double s_world[3]) const {
  if (body_id < 0) return false;
  int best_gi = -1;
  for (int gi = 0; gi < model->ngeom; ++gi) {
    if (model->geom_bodyid[gi] != body_id) continue;
    const char* gname = model->names + model->name_geomadr[gi];
    if (!gname || !*gname) continue;
    char boxname[256];
    std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
    int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
    if (bid >= 0) { best_gi = gi; break; }
  }
  if (best_gi < 0) return false;
  return BoxCenterRaySurfacePointForGeom(model, data, best_gi, p_world, s_world);
}

// Environment-only update: place/rotate mocap boxes without touching mode logic
void MjTwin::TransitionEnvOnlyLocked(mjModel* model, mjData* data) {
  // Lazy init mocap ids for boxes and foot geoms
  if (box_mocap_id_[0] < 0) {
    const char* box_names[4] = {"box_FL", "box_FR", "box_HL", "box_HR"};
    const char* foot_geom_names[4] = {"FL", "FR", "HL", "HR"};
    for (int i = 0; i < 4; ++i) {
      int bid = mj_name2id(model, mjOBJ_XBODY, box_names[i]);
      if (bid >= 0) box_mocap_id_[i] = model->body_mocapid[bid];
      foot_geom_id_boxref_[i] = mj_name2id(model, mjOBJ_GEOM, foot_geom_names[i]);
    }
    int g0 = mj_name2id(model, mjOBJ_GEOM, "box_FL_geom");
    if (g0 >= 0 && model->geom_type[g0] == mjGEOM_BOX) {
      box_half_height_ = model->geom_size[3 * g0 + 2];
    }
  }

  // Update per foot
  for (int i = 0; i < 4; ++i) {
    int mid = box_mocap_id_[i];
    int gid = foot_geom_id_boxref_[i];
    if (mid < 0 || gid < 0) continue;
    const double* pf = data->geom_xpos + 3 * gid;
    double s_world[3], n_world[3];
    double z_height;
    if (!terrain_.GetHeightFromWorld(data, pf[0], pf[1], z_height)) continue;
    s_world[0] = pf[0];
    s_world[1] = pf[1];
    s_world[2] = z_height;
    terrain_.GetNormalFromWorld(data, pf[0], pf[1], n_world);

    data->mocap_pos[3 * mid + 0] = s_world[0] - box_half_height_ * n_world[0];
    data->mocap_pos[3 * mid + 1] = s_world[1] - box_half_height_ * n_world[1];
    data->mocap_pos[3 * mid + 2] = s_world[2] - box_half_height_ * n_world[2];

    double z_axis[3] = {n_world[0], n_world[1], n_world[2]};
    double a[3] = {0.0, 0.0, 1.0};
    if (mju_abs(z_axis[2]) > 0.9) a[0] = 1.0, a[1] = 0.0, a[2] = 0.0;
    double x[3];
    mju_cross(x, a, z_axis);
    double nx = mju_norm3(x);
    if (nx < 1e-9) { x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; }
    else { mju_scl3(x, x, 1.0 / nx); }
    double y[3];
    mju_cross(y, z_axis, x);
    double R[9] = {x[0], y[0], z_axis[0], x[1], y[1], z_axis[1], x[2], y[2], z_axis[2]};
    double q[4];
    mju_mat2Quat(q, R);
    mju_copy(data->mocap_quat + 4 * mid, q, 4);
  }

  // Update generic pairs (other robot geoms to their designated boxes: trunk, knees, thighs, etc.)
  for (auto& e : generic_pairs_) {
    if (e.geom_id < 0 || e.mocap_id < 0) continue;
    const double* pg = data->geom_xpos + 3 * e.geom_id;
    double s_world[3], n_world[3];
    double z;
    if (!terrain_.GetHeightFromWorld(data, pg[0], pg[1], z)) continue;
    s_world[0] = pg[0];
    s_world[1] = pg[1];
    s_world[2] = z;
    terrain_.GetNormalFromWorld(data, pg[0], pg[1], n_world);

    // lazily fetch half-height from the mocap's attached geom size if unknown
    if (e.half_h <= 0.0) {
      const char* gname = model->names + model->name_geomadr[e.geom_id];
      char gbox[256];
      std::snprintf(gbox, sizeof(gbox), "box_%s_geom", gname);
      int gbox_id = mj_name2id(model, mjOBJ_GEOM, gbox);
      if (gbox_id >= 0 && model->geom_type[gbox_id] == mjGEOM_BOX) {
        e.half_h = model->geom_size[3 * gbox_id + 2];
      } else {
        e.half_h = box_half_height_;
      }
    }

    data->mocap_pos[3 * e.mocap_id + 0] = s_world[0] - e.half_h * n_world[0];
    data->mocap_pos[3 * e.mocap_id + 1] = s_world[1] - e.half_h * n_world[1];
    data->mocap_pos[3 * e.mocap_id + 2] = s_world[2] - e.half_h * n_world[2];

    double zn[3] = {n_world[0], n_world[1], n_world[2]};
    double an[3] = {0.0, 0.0, 1.0};
    if (mju_abs(zn[2]) > 0.9) an[0] = 1.0, an[1] = 0.0, an[2] = 0.0;
    double xn[3];
    mju_cross(xn, an, zn);
    double nxn = mju_norm3(xn);
    if (nxn < 1e-9) { xn[0] = 1.0; xn[1] = 0.0; xn[2] = 0.0; }
    else { mju_scl3(xn, xn, 1.0 / nxn); }
    double yn[3];
    mju_cross(yn, zn, xn);
    double Rn[9] = {xn[0], yn[0], zn[0], xn[1], yn[1], zn[1], xn[2], yn[2], zn[2]};
    double qn[4];
    mju_mat2Quat(qn, Rn);
    mju_copy(data->mocap_quat + 4 * e.mocap_id, qn, 4);
  }
}

}  // namespace mjpc
