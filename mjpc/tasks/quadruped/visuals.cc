#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/quadruped/mjTwin.h"
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"


namespace mjpc {
    
    
namespace { // Colors for visualization elements drawn in ModifyScene().
    constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};  // step-height cylinders
    constexpr float kHullRgba[4] = {0.4, 0.2, 0.8, 1};  // convex hull
    constexpr float kAvgRgba[4]  = {0.4, 0.2, 0.8, 1};   // average foot position
    constexpr float kCapRgba[4]  = {0.3, 0.3, 0.8, 1};   // capture point
    constexpr float kPcpRgba[4]  = {0.5, 0.5, 0.2, 1};   // projected capture point
}


// Draw task-related geometry in the scene.
void Quadruped::ModifyScene(const mjModel* model, const mjData* data, 
    mjvScene* scene) const {
        
    // current foot positions
    double* foot_pos[ResidualFn::kNumFoot];
    for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
        foot_pos[foot] = data->geom_xpos + 3 * residual_.foot_geom_id_[foot];
    }
    
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
            bool front_hand =
            !handstand && (foot == ResidualFn::kFootFL || foot == ResidualFn::kFootFR);
            bool back_hand =
            handstand && (foot == ResidualFn::kFootHL || foot == ResidualFn::kFootHR);
            if (front_hand || back_hand) continue;
        }
        if (step[foot]) {
            flight_pos[foot][2] =
            ResidualFn::kFootRadius + step[foot] + ground[foot];
            AddConnector(scene, mjGEOM_CYLINDER, ResidualFn::kFootRadius,
                stance_pos[foot], flight_pos[foot], kStepRgba);
            }
        }
        
        // support polygon (currently unused for cost)
        double polygon[2 * ResidualFn::kNumFoot];
        for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
            polygon[2 * foot] = foot_pos[foot][0];
            polygon[2 * foot + 1] = foot_pos[foot][1];
        }
        int hull[ResidualFn::kNumFoot];
        int num_hull = Hull2D(hull, ResidualFn::kNumFoot, polygon);
        for (int i = 0; i < num_hull; i++) {
            int j = (i + 1) % num_hull;
            AddConnector(scene, mjGEOM_CAPSULE, ResidualFn::kFootRadius / 2,
                stance_pos[hull[i]], stance_pos[hull[j]], kHullRgba);
            }
            
            // capture point
            bool is_biped = residual_.current_mode_ == ResidualFn::kModeBiped;
            double height_goal =
            is_biped ? ResidualFn::kHeightBiped : ResidualFn::kHeightQuadruped;
            
            // derive gravity magnitude on demand (flip-specific cache removed)
            double gravity = mju_norm3(model->opt.gravity);
            double fall_time = mju_sqrt(2 * height_goal / gravity);
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

static void VisualizeFootholdLogic(const mjModel* model, const mjData* data,
                                   mjvScene* scene,
                                   const Quadruped::ResidualFn& residual,
                                   const Terrain& terrain,
                                   const std::vector<double>& parameters) {
  const float rgba_nominal[4] = {0.2f, 0.6f, 1.0f, 0.35f};
  const float rgba_safe[4] = {0.2f, 0.9f, 0.2f, 0.9f};
  const float rgba_snap[4] = {0.9f, 0.4f, 0.1f, 0.9f};
  double ghost_radius = 0.02;
    double target_radius = 0.025;
  double snap_radius = 0.006;

  Quadruped::ResidualFn::A1Gait gait = residual.GetGait();
  double step[Quadruped::ResidualFn::kNumFoot];
  residual.FootStep(step, residual.GetPhase(data->time), gait);

  double* torso_pos = data->xipos + 3 * residual.torso_body_id_;
  double* goal_pos = data->mocap_pos + 3 * residual.goal_mocap_id_;

  double* foot_pos[Quadruped::ResidualFn::kNumFoot];
  for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
    foot_pos[foot] = data->geom_xpos + 3 * residual.foot_geom_id_[foot];
  }
  double* hip_pos[Quadruped::ResidualFn::kNumFoot];
  const double* hip_mat[Quadruped::ResidualFn::kNumFoot];
  for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
    hip_pos[foot] = data->xipos + 3 * residual.shoulder_body_id_[foot];
    hip_mat[foot] = data->xmat + 9 * residual.shoulder_body_id_[foot];
  }

  for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
    if (residual.current_mode_ == Quadruped::ResidualFn::kModeBiped) {
      bool handstand = ReinterpretAsInt(parameters[residual.biped_type_param_id_]);
      bool front_hand = !handstand && (foot == Quadruped::ResidualFn::kFootFL ||
                                       foot == Quadruped::ResidualFn::kFootFR);
      bool back_hand = handstand && (foot == Quadruped::ResidualFn::kFootHL ||
                                     foot == Quadruped::ResidualFn::kFootHR);
      if (front_hand || back_hand) continue;
    }

    // Visualize the same hip-anchored target (with 5cm torso +X + 10cm torso +Y offset)
    // that the cost uses.
    constexpr double kHipForwardOffset = 0.05;  // 5cm along torso +X
    const double* torso_xmat = data->xmat + 9 * residual.torso_body_id_;
    double torso_x[3] = {torso_xmat[0], torso_xmat[1], torso_xmat[2]};
    double lateral = (foot == Quadruped::ResidualFn::kFootFR ||
                      foot == Quadruped::ResidualFn::kFootHR) ? -0.10 : 0.10;
    double hip_offset_world[3] = {
        hip_mat[foot][3] * lateral,
        hip_mat[foot][4] * lateral,
        hip_mat[foot][5] * lateral};
    double query[3] = {hip_pos[foot][0] + hip_offset_world[0] + kHipForwardOffset * torso_x[0],
                       hip_pos[foot][1] + hip_offset_world[1] + kHipForwardOffset * torso_x[1],
                       hip_pos[foot][2] + hip_offset_world[2]};
    if (residual.current_mode_ == Quadruped::ResidualFn::kModeScramble) {
      double duty_ratio = parameters[residual.duty_param_id_];
      // Mirror query offset schedule from gait cost (aligned to StepHeight cylinders).
      double scale = 0.0;
      if (duty_ratio < 1.0) {
        double global_phase = residual.GetPhase(data->time);
        double foot_phase = 2 * mjPI * Quadruped::ResidualFn::kGaitPhase[gait][foot];
        double phi_full =
            std::fmod(global_phase - foot_phase + 2 * mjPI, 2 * mjPI) /
            (2 * mjPI);  // [0,1)

        const double half_swing = 0.5 * (1.0 - duty_ratio);
        const bool in_stance = (phi_full >= half_swing && phi_full <= 1.0 - half_swing);

        if (in_stance) {
          double stance_progress = (phi_full - half_swing) / duty_ratio;
          if (stance_progress < 0.50) {
            scale = 0.0;
          } else {
            double t = (stance_progress - 0.50) / 0.50;
            t = mju_clip(t, 0.0, 1.0);
            scale = t;
          }
        } else {
          double angle = fmod(global_phase + mjPI - foot_phase, 2 * mjPI) - mjPI;
          angle *= 0.5 / (1.0 - duty_ratio);
          angle = mju_clip(angle, -mjPI / 2, mjPI / 2);
          double swing_phase = (angle + mjPI / 2) / mjPI;

          if (swing_phase < 0.60) {
            scale = 1.0;
          } else {
            double t = (swing_phase - 0.60) / 0.40;
            t = mju_clip(t, 0.0, 1.0);
            scale = 1.0 - t;
          }
        }
      }

      double torso_to_goal[3];
      double* goal = goal_pos;
      mju_sub3(torso_to_goal, goal, torso_pos);
      mju_normalize3(torso_to_goal);
      mju_sub3(torso_to_goal, goal, hip_pos[foot]);
      torso_to_goal[2] = 0;
      mju_normalize3(torso_to_goal);
      mju_addToScl3(query, torso_to_goal, 0.15 * scale);
    }

    double nominal_ground = Ground(model, data, query);
    double nominal_pos[3] = {query[0], query[1], nominal_ground};

    double safe_xy[2];
    residual.GetProjectedFoothold(data, query, safe_xy);

    Terrain::PatchFeatures features{};
    terrain.GetPatchFeatures(data, safe_xy[0], safe_xy[1], features);

    // candidate footholds evaluated during rough-terrain search
    const double candidate_radii[] = {0.03, 0.05, 0.07, 0.09, 0.12};
    constexpr int kNumCandidates = 8;
    const float rgba_candidate[4] = {0.2f, 0.4f, 1.0f, 0.9f};
    const float rgba_bezier[4] = {0.9f, 0.4f, 0.1f, 0.9f};  // orange (trajectory)
    const float rgba_tracked[4] = {0.6f, 0.2f, 0.8f, 0.95f};  // purple (tracked point)
    double candidate_size[3] = {target_radius * 0.3, 0, 0};
    for (double rad : candidate_radii) {
      for (int i = 0; i < kNumCandidates; ++i) {
        double angle = 2.0 * mjPI * (static_cast<double>(i) / kNumCandidates);
        double cx = query[0] + rad * mju_cos(angle);
        double cy = query[1] + rad * mju_sin(angle);
        Terrain::PatchFeatures candidate_features{};
        terrain.GetPatchFeatures(data, cx, cy, candidate_features);
        double candidate_pos[3] = {cx, cy, candidate_features.max_height};
        AddGeom(scene, mjGEOM_SPHERE, candidate_size, candidate_pos,
                /*mat=*/nullptr, rgba_candidate);
      }
    }

    Terrain::PatchFeatures start_features{};
    double torso_y[3] = {torso_xmat[3], torso_xmat[4], torso_xmat[5]};
    // Match ComputeBezierTrajectory(): P0 is 5cm forward (torso +X) plus torso lateral offset.
    double P0_xy[2] = {hip_pos[foot][0] + kHipForwardOffset * torso_x[0] +
                           lateral * torso_y[0],
                       hip_pos[foot][1] + kHipForwardOffset * torso_x[1] +
                           lateral * torso_y[1]};
    terrain.GetPatchFeatures(data, P0_xy[0], P0_xy[1], start_features);

    double target_pos[3] = {safe_xy[0], safe_xy[1], features.max_height};

    double P3[3] = {safe_xy[0], safe_xy[1],
                    features.max_height + residual.kFootRadius};
    // Match ComputeBezierTrajectory(): P0 is 5cm forward (torso +X) plus torso lateral offset.
    double P0[3] = {P0_xy[0], P0_xy[1],
                    start_features.max_height + residual.kFootRadius};
    double target_amp = parameters[residual.amplitude_param_id_];

    double mid_x = 0.5 * (P0[0] + P3[0]);
    double mid_y = 0.5 * (P0[1] + P3[1]);
    Terrain::PatchFeatures mid_features{};
    terrain.GetPatchFeatures(data, mid_x, mid_y, mid_features);
    double midpoint_z = mid_features.max_height;

    double ground_ref = mju_max(
        mju_max(start_features.max_height, features.max_height), midpoint_z);

    // Adaptive clearance boost (mirror of ComputeBezierTrajectory).
    constexpr double kObstacleClearanceGain = 0.5;
    constexpr double kSlopeClearanceGain = 0.05;
    constexpr double kMaxExtraClearance = 0.10;
    double low_ref = mju_min(start_features.max_height, features.max_height);
    double obstacle_height = mju_max(0.0, ground_ref - low_ref);
    double min_normal_z = mju_min(
        mju_min(start_features.normal[2], features.normal[2]),
        mid_features.normal[2]);
    double slope_factor = mju_max(0.0, 1.0 - min_normal_z);
    double extra_clear =
        kObstacleClearanceGain * obstacle_height + kSlopeClearanceGain * slope_factor;
    extra_clear = mju_min(extra_clear, kMaxExtraClearance);

    double z_clear = ground_ref + residual.kFootRadius + target_amp + extra_clear;
    double P1[3] = {P0[0], P0[1], z_clear};
    double P2[3] = {P3[0], P3[1], z_clear};

    // Use StepHeight-aligned swing phase (matches CostRoughGround) for the tracked point and curve.
    double swing_phase = 0.0;
    double duty_ratio = parameters[residual.duty_param_id_];
    if (duty_ratio < 1.0) {
      double phase_now = residual.GetPhase(data->time);
      double footphase_now = 2 * mjPI * Quadruped::ResidualFn::kGaitPhase[gait][foot];
      double angle = fmod(phase_now + mjPI - footphase_now, 2 * mjPI) - mjPI;
      angle *= 0.5 / (1.0 - duty_ratio);
      angle = mju_clip(angle, -mjPI / 2, mjPI / 2);
      swing_phase = (angle + mjPI / 2) / mjPI;
    }
    auto EvalBezier = [&](double t, double out[3]) {
      double o = 1.0 - t;
      double o2 = o * o;
      double t2 = t * t;
      double bb0 = o2 * o;
      double bb1 = 3 * o2 * t;
      double bb2 = 3 * o * t2;
      double bb3 = t * t2;
      for (int i = 0; i < 3; ++i) {
        out[i] = bb0 * P0[i] + bb1 * P1[i] + bb2 * P2[i] + bb3 * P3[i];
      }
    };

    // Tracked reference point (purple sphere, twice as large as candidates).
    double tracked[3];
    EvalBezier(swing_phase, tracked);
    double tracked_size[3] = {candidate_size[0] * 2.0, 0, 0};
    AddGeom(scene, mjGEOM_SPHERE, tracked_size, tracked, /*mat=*/nullptr,
            rgba_tracked);

    // Draw full Bezier trajectory as a thin orange polyline.
    constexpr int kBezierSamples = 12;
    double prev[3];
    EvalBezier(0.0, prev);
    for (int i = 1; i <= kBezierSamples; ++i) {
      double t = static_cast<double>(i) / kBezierSamples;
      double curr[3];
      EvalBezier(t, curr);
      AddConnector(scene, mjGEOM_CAPSULE, snap_radius, prev, curr, rgba_bezier);
      mju_copy3(prev, curr);
    }

    double ghost_size[3] = {ghost_radius, 0, 0};
    AddGeom(scene, mjGEOM_SPHERE, ghost_size, nominal_pos, /*mat=*/nullptr,
            rgba_nominal);

    double target_size[3] = {target_radius, 0, 0};
    AddGeom(scene, mjGEOM_SPHERE, target_size, target_pos, /*mat=*/nullptr,
            rgba_safe);

    AddConnector(scene, mjGEOM_CAPSULE, snap_radius, nominal_pos, target_pos,
                 rgba_snap);
  }
}

// static void VisualizeTerrainNormals(const mjData* data, mjvScene* scene,
//                              const Terrain& terrain) {
//   double gpos[3];
//   const double* gpos_src = data->geom_xpos + 3 * terrain.geom_id;
//   gpos[0] = gpos_src[0];
//   gpos[1] = gpos_src[1] - 2.0;
//   gpos[2] = gpos_src[2];
//   const double* gmat = data->geom_xmat + 9 * terrain.geom_id;

//   int W = terrain.ncol;
//   int H = terrain.nrow;
//   if (W < 4 || H < 4) return;
//   int col0 = (W - 4) / 2;
//   int row0 = (H - 4) / 2;

//   const float rgba[4] = {1.0f, 0.1f, 0.1f, 1.0f};
//   double arrow_radius = 0.02;
//   double arrow_len = 0.30;
//   double base_radius = 0.015;

//   for (int di = 0; di < 4; ++di) {
//     for (int dj = 0; dj < 4; ++dj) {
//       int col = col0 + dj;
//       int row = row0 + di;
//       double nloc[3];
//       terrain.GetNormalFromLocal(-terrain.sx + col * terrain.dx,
//                                  -terrain.sy + row * terrain.dy, nloc);

//       double x_local = -terrain.sx + col * terrain.dx;
//       double y_local = -terrain.sy + row * terrain.dy;
//       double z_local = 0.0;
//       terrain.GetHeightFromLocal(x_local, y_local, z_local);

//       double p_local[3] = {x_local, y_local, z_local};
//       double p_world[3];
//       double n_world[3];
//       mju_mulMatVec(p_world, gmat, p_local, 3, 3);
//       mju_mulMatVec(n_world, gmat, nloc, 3, 3);

//       p_world[0] += gpos[0];
//       p_world[1] += gpos[1];
//       p_world[2] += gpos[2];

//       double tip_world[3] = {p_world[0] + arrow_len * n_world[0],
//                              p_world[1] + arrow_len * n_world[1],
//                              p_world[2] + arrow_len * n_world[2]};

//       double base_size[3] = {base_radius, 0, 0};
//       AddGeom(scene, mjGEOM_SPHERE, base_size, p_world, /*mat=*/nullptr, rgba);
//       AddConnector(scene, mjGEOM_CAPSULE, arrow_radius, p_world, tip_world,
//                    rgba);
//     }
//   }

//   double x_min_local = -terrain.sx + col0 * terrain.dx;
//   double x_max_local = -terrain.sx + (col0 + 3) * terrain.dx;
//   double y_min_local = -terrain.sy + row0 * terrain.dy;
//   double y_max_local = -terrain.sy + (row0 + 3) * terrain.dy;

//   auto HeightBilinearLocal = [&](double x_local, double y_local) {
//     double z_local = 0.0;
//     terrain.GetHeightFromLocal(x_local, y_local, z_local);
//     return z_local;
//   };

//   const float rgba_i[4] = {0.1f, 1.0f, 1.0f, 1.0f};
//   double arrow_radius_i = 0.012;
//   double arrow_len_i = 0.20;
//   double base_radius_i = 0.010;

//   int S = 7;
//   for (int iy = 0; iy < S; ++iy) {
//     double ty = (S == 1) ? 0.0 : static_cast<double>(iy) / static_cast<double>(S - 1);
//     double y_loc = (1.0 - ty) * y_min_local + ty * y_max_local;
//     for (int ix = 0; ix < S; ++ix) {
//       double tx = (S == 1) ? 0.0 : static_cast<double>(ix) / static_cast<double>(S - 1);
//       double x_loc = (1.0 - tx) * x_min_local + tx * x_max_local;

//       double n_loc[3];
//       terrain.GetNormalFromLocal(x_loc, y_loc, n_loc);
//       double z_loc = HeightBilinearLocal(x_loc, y_loc);

//       double p_loc[3] = {x_loc, y_loc, z_loc};
//       double p_w[3], n_w[3];
//       mju_mulMatVec(p_w, gmat, p_loc, 3, 3);
//       mju_mulMatVec(n_w, gmat, n_loc, 3, 3);
//       p_w[0] += gpos[0];
//       p_w[1] += gpos[1];
//       p_w[2] += gpos[2];

//       double tip_w[3] = {p_w[0] + arrow_len_i * n_w[0],
//                          p_w[1] + arrow_len_i * n_w[1],
//                          p_w[2] + arrow_len_i * n_w[2]};

//       double base_size_i[3] = {base_radius_i, 0, 0};
//       AddGeom(scene, mjGEOM_SPHERE, base_size_i, p_w, /*mat=*/nullptr, rgba_i);
//       AddConnector(scene, mjGEOM_CAPSULE, arrow_radius_i, p_w, tip_w, rgba_i);
//     }
//   }
// }

static void VisualizeBodyClearance(const mjModel* model, const mjData* data,
                            mjvScene* scene,
                            const Quadruped::ResidualFn& residual) {
  const float rgba_head[4] = {1.0f, 0.2f, 0.2f, 1.0f};
  const float rgba_knee[4] = {1.0f, 0.2f, 0.2f, 1.0f};
  double site_r = 0.03;
  double sz3[3] = {site_r, 0, 0};

  if (residual.head_site_id_ >= 0) {
    const double* p = data->site_xpos + 3 * residual.head_site_id_;
    int trunk_bid = mj_name2id(model, mjOBJ_BODY, "trunk");
    double head_r = site_r;
    if (trunk_bid >= 0) {
      int best = -1;
      double bestd2 = 1e30;
      for (int gi = 0; gi < model->ngeom; ++gi) {
        if (model->geom_type[gi] != mjGEOM_SPHERE) continue;
        if (model->geom_bodyid[gi] != trunk_bid) continue;
        if (model->geom_group[gi] != 3) continue;
        const double* gc = data->geom_xpos + 3 * gi;
        double dx = gc[0] - p[0];
        double dy = gc[1] - p[1];
        double dz = gc[2] - p[2];
        double d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < bestd2) {
          bestd2 = d2;
          best = gi;
        }
      }
      if (best >= 0) head_r = model->geom_size[3 * best + 0];
    }
    double sz_head[3] = {head_r, 0, 0};
    AddGeom(scene, mjGEOM_SPHERE, sz_head, p, /*mat=*/nullptr, rgba_head);
  }

  for (int k = 0; k < 4; ++k) {
    int bid = residual.knee_body_id_[k];
    if (bid < 0) continue;
    const double* p = data->xpos + 3 * bid;
    AddGeom(scene, mjGEOM_SPHERE, sz3, p, /*mat=*/nullptr, rgba_knee);
  }
}

void mjpc::MjTwin::ModifyScene(const mjModel* model, const mjData* data,
                               mjvScene* scene) const {

    Quadruped::ModifyScene(model, data, scene);

    /* ------ Debugging the new foothold mechanism -------- */
    if (terrain_.geom_id >= 0 && residual_.terrain_) {
        VisualizeFootholdLogic(model, data, scene, residual_, terrain_, parameters);
        // VisualizeTerrainNormals(data, scene, terrain_);
        VisualizeBodyClearance(model, data, scene, residual_);
    }
}

}