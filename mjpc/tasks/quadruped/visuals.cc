#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/quadruped/mjTwin.h"
#include "mjpc/tasks/quadruped/footholds.h"
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"


namespace mjpc {
    
    
namespace { // Colors for visualization elements drawn in ModifyScene().
    constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};  // step-height cylinders
    constexpr float kHullRgba[4] = {0.4, 0.2, 0.8, 1};  // convex hull
    constexpr float kAvgRgba[4]  = {0.4, 0.2, 0.8, 1};   // average foot position
    constexpr float kCapRgba[4]  = {0.3, 0.3, 0.8, 1};   // capture point
    constexpr float kPcpRgba[4]  = {0.5, 0.5, 0.2, 1};   // projected capture point
    constexpr float kQueryRgba[4] = {0.1f, 0.5f, 1.0f, 0.7f};   // nominal query
    constexpr float kCandidateRgba[4] = {0.1f, 0.5f, 1.0f, 0.4f}; // sampled candidates
    constexpr float kChosenRgba[4] = {0.1f, 0.9f, 0.2f, 0.9f};  // chosen landing
}


// Draw task-related geometry in the scene.
void Quadruped::ModifyScene(const mjModel* model, const mjData* data, 
    mjvScene* scene) const {
        
    // current foot positions
    double* foot_pos[ResidualFn::kNumFoot];
    for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
        int gid = residual_.foot_geom_id_[foot];
        if (gid < 0) return;  // not initialized yet (e.g., during reload/reset)
        foot_pos[foot] = data->geom_xpos + 3 * gid;
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
  const float rgba_bezier[4] = {0.9f, 0.4f, 0.1f, 0.9f};    // orange curve
  const float rgba_tracked[4] = {0.6f, 0.2f, 0.8f, 0.95f}; // purple point
  double snap_radius = 0.006;
  double tracked_radius = 0.02;
  double query_radius = 0.015;
  double candidate_radius = 0.008;

  // read current targets (point on curve)
  double* foothold_targets = SensorByName(model, data, "foothold_targets");

  Quadruped::ResidualFn::A1Gait gait = residual.GetGait();
  double step[Quadruped::ResidualFn::kNumFoot];
  residual.FootStep(step, residual.GetPhase(data->time), gait);
  double duty_ratio = parameters[residual.duty_param_id_];

  const double* torso_mat = data->xmat + 9 * residual.torso_body_id_;
  // Use torso x-axis column as forward (ignore z component).
  double torso_x[3] = {torso_mat[0], torso_mat[3], 0.0};
  mju_normalize3(torso_x);

  for (Quadruped::ResidualFn::A1Foot foot : Quadruped::ResidualFn::kFootAll) {
    // skip "hands" in biped mode
    if (residual.current_mode_ == Quadruped::ResidualFn::kModeBiped) {
      bool handstand = ReinterpretAsInt(parameters[residual.biped_type_param_id_]);
      bool front_hand = !handstand && (foot == Quadruped::ResidualFn::kFootFL ||
                                       foot == Quadruped::ResidualFn::kFootFR);
      bool back_hand = handstand && (foot == Quadruped::ResidualFn::kFootHL ||
                                     foot == Quadruped::ResidualFn::kFootHR);
      if (front_hand || back_hand) continue;
    }

    // rely on latched control points from footholds; only draw when Bezier is active
    const FootholdPlanner* planner = residual.foothold_planner_;
    const bool bezier_active = planner && planner->bezier_active_[foot];

    // reconstruct nominal query (15cm forward) and drop to hfield
    const double* p0 = data->geom_xpos + 3 * residual.foot_geom_id_[foot];
    double query[3];
    mju_copy3(query, p0);
    mju_addToScl3(query, torso_x, 0.15);
    double qh = Ground(model, data, query);
    query[2] = qh;

    // visualize nominal query (blue) at ground
    double qsize[3] = {query_radius, 0, 0};
    AddGeom(scene, mjGEOM_SPHERE, qsize, query, /*mat=*/nullptr, kQueryRgba);

    // safety sampling: forward points on hfield
    bool unsafe = false;
    const double forward_offsets[5] = {0.03, 0.06, 0.09, 0.12, 0.15};
    double sample_size[3] = {candidate_radius, 0, 0};
    for (double off : forward_offsets) {
      double sample[3];
      mju_scl3(sample, torso_x, off);
      mju_addTo3(sample, p0);
      sample[2] = Ground(model, data, sample);
      AddGeom(scene, mjGEOM_SPHERE, sample_size, sample, /*mat=*/nullptr, kCandidateRgba);
      if (residual.terrain_ && !residual.terrain_->IsSafe(data, sample[0], sample[1])) {
        unsafe = true;
      }
    }

    double chosen[3];
    mju_copy3(chosen, query);
    if (unsafe && residual.terrain_) {
      const double radii[] = {0.03, 0.05, 0.07, 0.09, 0.12};
      constexpr int kNumCandidates = 8;
      double best_dist2 = std::numeric_limits<double>::infinity();
      for (double rad : radii) {
        for (int i = 0; i < kNumCandidates; ++i) {
          double ang = 2.0 * mjPI * (static_cast<double>(i) / kNumCandidates);
          double cx = query[0] + rad * mju_cos(ang);
          double cy = query[1] + rad * mju_sin(ang);
          double candidate[3] = {cx, cy, query[2]};
          candidate[2] = Ground(model, data, candidate);
          AddGeom(scene, mjGEOM_SPHERE, sample_size, candidate, /*mat=*/nullptr, kCandidateRgba);
          if (residual.terrain_->IsSafe(data, cx, cy)) {
            double dx = cx - query[0];
            double dy = cy - query[1];
            double d2 = dx * dx + dy * dy;
            if (d2 < best_dist2) {
              best_dist2 = d2;
              chosen[0] = cx;
              chosen[1] = cy;
              chosen[2] = candidate[2];
            }
          }
        }
      }
    }

    // chosen point (green) on ground
    AddGeom(scene, mjGEOM_SPHERE, qsize, chosen, /*mat=*/nullptr, kChosenRgba);

    if (!bezier_active) continue;  // no-bezier: no curve

    // swing phase aligned with step-height cylinders
    double phase = residual.GetPhase(data->time);
    double footphase = 2 * mjPI * Quadruped::ResidualFn::kGaitPhase[gait][foot];
    double swing_phase = FootholdPlanner::SwingPhase(phase, footphase, duty_ratio);

    if (!planner) continue;

    // draw full Bezier trajectory
    constexpr int kSamples = 12;
    double prev[3];
    planner->EvalBezier(foot, 0.0, prev);
    for (int i = 1; i <= kSamples; ++i) {
      double t = static_cast<double>(i) / kSamples;
      double curr[3];
      planner->EvalBezier(foot, t, curr);
      AddConnector(scene, mjGEOM_CAPSULE, snap_radius, prev, curr, rgba_bezier);
      mju_copy3(prev, curr);
    }

    // tracked point: either from sensor or evaluated Bezier at swing phase
    double tracked[3];
    if (foothold_targets) {
      const double* src = foothold_targets + 3 * foot;
      if (src[0] == 0.0 && src[1] == 0.0 && src[2] == 0.0 && planner) {
        // If the sensor data hasn't been updated yet, fall back to the curve
        // to avoid a flicker at the origin.
        planner->EvalBezier(foot, swing_phase, tracked);
      } else {
        mju_copy3(tracked, src);
      }
    } else {
      planner->EvalBezier(foot, swing_phase, tracked);
    }
    double tracked_size[3] = {tracked_radius, 0, 0};
    AddGeom(scene, mjGEOM_SPHERE, tracked_size, tracked, /*mat=*/nullptr,
            rgba_tracked);
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