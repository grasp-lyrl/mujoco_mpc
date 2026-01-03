#include "mjpc/tasks/quadruped/mjTwin.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/tasks/quadruped/terrain.h"
#include "mjpc/tasks/quadruped/footholds.h"
#include "mjpc/utilities.h"


namespace mjpc {

std::string MjTwin::XmlPath() const { return GetModelPath("quadruped/xmls/task_mjTwin.xml"); }
std::string MjTwin::Name() const { return "mjTwin"; }
  
void MjTwin::ResetLocked(const mjModel* model) {
  
    Quadruped::ResetLocked(model);
    
    terrain_.Initialize(model);
    residual_.foothold_planner_ = &foothold_planner_;

    unsafe_hfield_id_ = mj_name2id(model, mjOBJ_HFIELD, "hf133_unsafe");
    unsafe_hfield_dirty_ = true;
    if (unsafe_hfield_id_ >= 0) {
        mjModel* mutable_model = const_cast<mjModel*>(model);
        mjData* temp_data = mj_makeData(mutable_model);
        if (temp_data) {
            mj_forward(mutable_model, temp_data);
            UpdateUnsafeVisualizationHField(mutable_model, temp_data);
            mj_deleteData(temp_data);
        }
    }
  
    int box_FL_geom_id = mj_name2id(model, mjOBJ_GEOM, "box_FL_geom");
    if (box_FL_geom_id < 0 || model->geom_type[box_FL_geom_id] != mjGEOM_BOX) {
        mju_error("box_FL_geom missing or not a box; error in collisions.xml");
    }
    mju_copy3(box_half_size_, model->geom_size + 3 * box_FL_geom_id);
    mju_copy3(residual_.box_half_size_, box_half_size_);
    residual_.terrain_ = &terrain_;
  
    // cache foot collision box mocap ids
    const char* box_names[4] = {"box_FL", "box_HL", "box_FR", "box_HR"};
    for (A1Foot foot : kFootAll) {
        int bid = mj_name2id(model, mjOBJ_XBODY, box_names[foot]);
        if (bid >= 0) {
            box_mocap_id_[foot] = model->body_mocapid[bid];
        }
    }
    
    collision_pairs_.clear();
    for (int gi = 0; gi < model->ngeom; ++gi) {
        
        const char* gname = mj_id2name(model, mjOBJ_GEOM, gi);
        if (!gname || !*gname)
        continue;

    char boxname[256];
    std::snprintf(boxname, sizeof(boxname), "box_%s", gname);
    int bid = mj_name2id(model, mjOBJ_XBODY, boxname);
        if (bid < 0)
            continue;

        int box_id = model->body_mocapid[bid];
        if (box_id < 0)
            continue;

        collision_pairs_.push_back({gi, box_id});
    }
}


void MjTwin::TransitionLocked(mjModel* model, mjData* data) {
    Quadruped::TransitionLocked(model, data);
    double duty_ratio = parameters[residual_.duty_param_id_];
    foothold_planner_.ComputeFootholds(model, data, residual_, duty_ratio);
    UpdateCollisionBoxes(model, data);
    UpdateUnsafeVisualizationHField(model, data);
  }
    
void MjTwin::TransitionEnvOnlyLocked(mjModel* model, mjData* data) {
    UpdateCollisionBoxes(model, data);
    UpdateUnsafeVisualizationHField(model, data);
} 
    

/* Place and orient all collision boxes under the respective geoms */
void MjTwin::UpdateCollisionBoxes(const mjModel* model, mjData* data) {

    double position[3], normal[3], q[4];

    for (auto& pair : collision_pairs_) {

        const double* geom_pos = data->geom_xpos + 3 * pair.geom_id;

        position[0] = geom_pos[0];
        position[1] = geom_pos[1];
        terrain_.GetHeightFromWorld(data, geom_pos[0], geom_pos[1], position[2]);
        terrain_.GetNormalFromWorld(data, geom_pos[0], geom_pos[1], normal);

        // position
        mju_addScl3(data->mocap_pos + 3 * pair.box_id, position, normal, -box_half_size_[2]);

        // orientation
        mju_quatZ2Vec(q, normal);
        mju_copy(data->mocap_quat + 4 * pair.box_id, q, 4);
    }
}

void MjTwin::UpdateUnsafeVisualizationHField(mjModel* model,
                                             const mjData* data) {
    if (unsafe_hfield_id_ < 0 || terrain_.hfield_id < 0 || !unsafe_hfield_dirty_) {
        return;
    }

    // validate dimensions match the physical terrain hfield
    int terrain_rows = model->hfield_nrow[terrain_.hfield_id];
    int terrain_cols = model->hfield_ncol[terrain_.hfield_id];
    if (terrain_rows != model->hfield_nrow[unsafe_hfield_id_] ||
        terrain_cols != model->hfield_ncol[unsafe_hfield_id_]) {
        mju_error("unsafe visualization hfield dims mismatch terrain");
        return;
    }

    const float* terrain_data = model->hfield_data + model->hfield_adr[terrain_.hfield_id];
    float* unsafe_data = model->hfield_data + model->hfield_adr[unsafe_hfield_id_];

    double terrain_scale = model->hfield_size[4 * terrain_.hfield_id + 2];
    double unsafe_scale = model->hfield_size[4 * unsafe_hfield_id_ + 2];

    const double* R = data->geom_xmat + 9 * terrain_.geom_id;
    const double* t = data->geom_xpos + 3 * terrain_.geom_id;

    for (int r = 0; r < terrain_rows; ++r) {
        double y_local = -terrain_.sy + r * terrain_.dy;
        for (int c = 0; c < terrain_cols; ++c) {
            double x_local = -terrain_.sx + c * terrain_.dx;
            int idx = r * terrain_cols + c;

            // world position at the cell center on the terrain surface (z unused)
            // convert cell center from terrain-local to world (xmat is row-major)
            double p_world[3] = {
                R[0] * x_local + R[1] * y_local + t[0],
                R[3] * x_local + R[4] * y_local + t[1],
                R[6] * x_local + R[7] * y_local + t[2]
            };

            bool is_safe = terrain_.IsSafe(data, p_world[0], p_world[1]);
            double base_height = static_cast<double>(terrain_data[idx]) * terrain_scale;

            double desired_height = base_height;
            if (!is_safe) desired_height = base_height + 0.06;  // -> (base + 0.10) - 0.05 = base + 0.05 world

            desired_height = mju_max(0.0, desired_height);
            unsafe_data[idx] = static_cast<float>(desired_height / unsafe_scale);
        }
    }

    unsafe_hfield_dirty_ = false;
}

} // namespace mjpc
