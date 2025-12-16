#ifndef MJPC_TASKS_QUADRUPED_MJTWIN_H_
#define MJPC_TASKS_QUADRUPED_MJTWIN_H_


#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/quadruped/terrain.h"


namespace mjpc {
    
class MjTwin : public Quadruped {

    public:

        std::string Name() const override;
        std::string XmlPath() const override;

        void ResetLocked(const mjModel* model) override;
        void TransitionLocked(mjModel* model, mjData* data) override;
        void TransitionEnvOnlyLocked(mjModel* model, mjData* data) override;

        void ModifyScene(const mjModel* model, const mjData* data,
            mjvScene* scene) const override;

        Terrain terrain_;
        
    private:

        void UpdateCollisionBoxes(const mjModel* model, mjData* data);
        void UpdateUnsafeVisualizationHField(mjModel* model,
                                             const mjData* data);
        
        int box_mocap_id_[4] = {-1, -1, -1, -1};
        double box_half_size_[3] = {0.0, 0.0, 0.0};

        int unsafe_hfield_id_ = -1;
        bool unsafe_hfield_dirty_ = true;
        
        struct CollisionPair {
            int geom_id = -1;
            int box_id  = -1;
        };
        std::vector<CollisionPair> collision_pairs_;

        enum A1Foot {
            kFootFL  = 0,
            kFootHL,
            kFootFR,
            kFootHR,
            kNumFoot
        };
        constexpr static A1Foot kFootAll[kNumFoot] = {kFootFL, kFootHL, kFootFR, kFootHR};
    };
    
}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_MJTWIN_H_
                            