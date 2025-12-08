#include "mjpc/tasks/quadruped/visuals.h"
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


void mjpc::MjTwin::ModifyScene(const mjModel* model, const mjData* data,
                               mjvScene* scene) const {

    Quadruped::ModifyScene(model, data, scene);

    /* ------------------------------------------------------------ */
    /* anything from this point onwards is debugging visualizations */
    /* ------------------------------------------------------------ */

    // terrain geom pose
    const double* gpos = data->geom_xpos + 3 * terrain_.geom_id;
    const double* gmat = data->geom_xmat + 9 * terrain_.geom_id; 

    // visualize 16 vertex normals in a central 4x4 grid
    int W = terrain_.ncol;
    int H = terrain_.nrow;
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
            double nloc[3];
            terrain_.GetNormalFromLocal(-terrain_.sx + col * terrain_.dx,
                                        -terrain_.sy + row * terrain_.dy,
                                        nloc);

            // local vertex position in geom frame
            double x_local = -terrain_.sx + col * terrain_.dx;
            double y_local = -terrain_.sy + row * terrain_.dy;

            double z_local = 0.0;
            terrain_.GetHeightFromLocal(x_local, y_local, z_local);

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
    double x_min_local = -terrain_.sx + col0 * terrain_.dx;
    double x_max_local = -terrain_.sx + (col0 + 3) * terrain_.dx;
    double y_min_local = -terrain_.sy + row0 * terrain_.dy;
    double y_max_local = -terrain_.sy + (row0 + 3) * terrain_.dy;

    // Bilinear height sampler in local frame
    auto HeightBilinearLocal = [&](double x_local, double y_local) {
        double z_local = 0.0;
        terrain_.GetHeightFromLocal(x_local, y_local, z_local);
        return z_local;
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
            terrain_.GetNormalFromLocal(x_loc, y_loc, n_loc);
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

}