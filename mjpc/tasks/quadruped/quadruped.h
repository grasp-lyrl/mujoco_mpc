#ifndef MJPC_TASKS_QUADRUPED_QUADRUPED_H_
#define MJPC_TASKS_QUADRUPED_QUADRUPED_H_

#include <string>
#include <vector>
#include <atomic>
#include <cstddef>
#include <cstring>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/quadruped/terrain.h"


namespace mjpc {

class Quadruped : public Task {
  public:
    std::string Name() const override;
    std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Quadruped* task) : mjpc::BaseResidualFn(task) {}
    ResidualFn(const ResidualFn&) = default;
    void Residual(const mjModel* model, const mjData* data, double* residual) const override;

    // default position cost weight (shared across tasks)
    constexpr static double kPositionWeightDefault = 0.24;

   private:
    friend class Quadruped;    
    //  ============  enums  ============
    // modes
    enum A1Mode {
      kModeQuadruped = 0,
      kModeBiped,
      kModeWalk,
      kModeScramble,
      kNumMode
    };

    // feet
    enum A1Foot {
      kFootFL  = 0,
      kFootHL,
      kFootFR,
      kFootHR,
      kNumFoot
    };

    // gaits
    enum A1Gait {
      kGaitStand = 0,
      kGaitWalk,
      kGaitTrot,
      kGaitCanter,
      kGaitGallop,
      kNumGait
    };

    //  ============  constants  ============
    constexpr static A1Foot kFootAll[kNumFoot] = {kFootFL, kFootHL,
                                                  kFootFR, kFootHR};
    constexpr static A1Foot kFootHind[2] = {kFootHL, kFootHR};
    constexpr static A1Gait kGaitAll[kNumGait] = {kGaitStand, kGaitWalk,
                                                  kGaitTrot, kGaitCanter,
                                                  kGaitGallop};

    // gait phase signature (normalized)
    constexpr static double kGaitPhase[kNumGait][kNumFoot] =
    {
    // FL     HL     FR     HR
      {0,     0,     0,     0   },   // stand
      {0,     0.75,  0.5,   0.25},   // walk
      {0,     0.5,   0.5,   0   },   // trot
      {0,     0.33,  0.33,  0.66},   // canter
      {0,     0.4,   0.05,  0.35}    // gallop
    };

    // gait parameters, set when switching into gait
    constexpr static double kGaitParam[kNumGait][6] =
    {
    // duty ratio  cadence  amplitude  balance   upright   height
    // unitless    Hz       meter      unitless  unitless  unitless
      {1,          1,       0,         0.3,      1,        1},      // stand
      {0.75,       1,       0.03,      0,        1,        1},      // walk
      {0.45,       2,       0.03,      0.2,      1,        1},      // trot
      {0.4,        4,       0.05,      0.03,     0.5,      0.2},    // canter
      {0.3,        3.5,     0.10,      0.03,     0.2,      0.1}     // gallop
    };

    // velocity ranges for automatic gait switching, meter/second
    constexpr static double kGaitAuto[kNumGait] =
    {
      0,     // stand
      0.02,  // walk
      0.02,  // trot
      0.6,   // canter
      2,     // gallop
    };
    // notes:
    // - walk is never triggered by auto-gait
    // - canter actually has a wider range than gallop

    // automatic gait switching: time constant for com speed filter
    constexpr static double kAutoGaitFilter = 0.2;    // second

    // automatic gait switching: minimum time between switches
    constexpr static double kAutoGaitMinTime = 1;     // second

    // target torso height over feet when quadrupedal
    // constexpr static double kHeightQuadruped = 0.25;  // meter
    constexpr static double kHeightQuadruped = 0.27;  // meter

    // target torso height over feet when bipedal
    constexpr static double kHeightBiped = 0.50;       // meter

    // radius of foot geoms
    constexpr static double kFootRadius = 0.02;       // meter

    // below this target yaw velocity, walk straight
    constexpr static double kMinAngvel = 0.01;        // radian/second

    // posture gain factors for abduction, hip, knee
    constexpr static double kJointPostureGain[3] = {2, 1, 1};  // unitless

    //  ============  methods  ============
    // return internal phase clock
    double GetPhase(double time) const;

    // return current gait
    A1Gait GetGait() const;

    // compute average foot position, depending on mode
    void AverageFootPos(double avg_foot_pos[3],
                        double* foot_pos[kNumFoot]) const;

    // return normalized target step height
    double StepHeight(double time, double footphase, double duty_ratio) const;

    // compute target step height for all feet
    void FootStep(double step[kNumFoot], double time, A1Gait gait) const;

    // walk horizontal position given time
    void Walk(double pos[2], double time) const;

    //  ============  task state variables, managed by Transition  ============
    A1Mode current_mode_       = kModeQuadruped;
    double last_transition_time_ = -1;

    // common mode states
    double mode_start_time_  = 0;
    double position_[3]       = {0};

    // walk states
    double heading_[2]        = {0};
    double speed_             = 0;
    double angvel_            = 0;

    // backflip states
    // gait-related states
    double current_gait_      = kGaitStand;
    double phase_start_       = 0;
    double phase_start_time_  = 0;
    double phase_velocity_    = 0;
    double com_vel_[2]        = {0};
    double gait_switch_time_  = 0;

    //  ============  constants, computed in Reset()  ============
    int torso_body_id_        = -1;
    int head_site_id_         = -1;
    int goal_mocap_id_        = -1;
    int gait_param_id_        = -1;
    int gait_switch_param_id_ = -1;
    int biped_type_param_id_  = -1;
    int cadence_param_id_     = -1;
    int amplitude_param_id_   = -1;
    int duty_param_id_        = -1;
    int arm_posture_param_id_ = -1;
    int upright_cost_id_      = -1;
    int balance_cost_id_      = -1;
    int height_cost_id_       = -1;
    int foot_geom_id_[kNumFoot];
    int shoulder_body_id_[kNumFoot];

    // terrain references for smooth stance gating
    int terrain_hfield_id_    = -1;
    int terrain_geom_id_      = -1;

    // clearance cost: ids and radii
    int clear_cost_id_        = -1;
    int head_site_id_clear_   = -1;
    int knee_body_id_clear_[4] = {-1, -1, -1, -1};  // FL, FR, HL, HR
    double head_radius_clear_ = 0.03;
    double knee_radius_clear_ = 0.03;
    // trunk forward sensors to penalize (match collision meshes):
    // cylinder and sphere geoms on trunk (group=3)
    int trunk_cyl_geom_id_clear_ = -1;
    int trunk_sph_geom_id_clear_ = -1;
    double trunk_cyl_radius_clear_ = 0.03;  // cylinder radius (size[0])
    double trunk_sph_radius_clear_ = 0.03;  // sphere radius (size[0])

  };

  Quadruped() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

  // call base-class Reset, save task-related ids
  void ResetLocked(const mjModel* model) override;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }
  // Some variants omit cost sensors; allow them to skip gracefully.
  virtual bool AllowMissingCostTerms() const { return false; }
  // Some variants do not use quadruped gait parameters at all.
  virtual bool UseGaitParameters() const { return true; }

 private:
  friend class ResidualFn;
  ResidualFn residual_;
};


class MjTwin : public Quadruped {
 public:

  std::string Name() const override;
  std::string XmlPath() const override;

  void ResetLocked(const mjModel* model) override;
  void TransitionLocked(mjModel* model, mjData* data) override;
  void TransitionEnvOnlyLocked(mjModel* model, mjData* data) override;
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;
  Terrain& terrain() { return terrain_; }
  const Terrain& terrain() const { return terrain_; }

  // Return the top surface point and normal (world frame) of the mocap box
  // corresponding to a given robot collision geom. Returns false if no such
  // box mapping exists.
  bool BoxTopSurfaceAndNormalForGeom(const mjModel* model, const mjData* data,
                                     int geom_id,
                                     double s_world[3], double n_world[3]) const;

  // Same as above, but resolves via any collision geom that belongs to the
  // specified body. Returns false if no mapped geom is found.
  bool BoxTopSurfaceAndNormalForBody(const mjModel* model, const mjData* data,
                                     int body_id,
                                     double s_world[3], double n_world[3]) const;

  // Intersection of the line from mesh point to box center with the box surface.
  // Returns false if no corresponding box is found. s_world is the surface point
  // where the line (from box center towards mesh point) exits the box.
  bool BoxCenterRaySurfacePointForGeom(const mjModel* model, const mjData* data,
                                       int geom_id, const double p_world[3],
                                       double s_world[3]) const;

  bool BoxCenterRaySurfacePointForBody(const mjModel* model, const mjData* data,
                                       int body_id, const double p_world[3],
                                       double s_world[3]) const;

  // Closest point on the surface of the mocap box to a given world point.
  // Uses oriented box geometry; returns false if no box is mapped.
  bool BoxClosestSurfacePointForGeom(const mjModel* model, const mjData* data,
                                     int geom_id, const double p_world[3],
                                     double s_world[3]) const;
  bool BoxClosestSurfacePointForBody(const mjModel* model, const mjData* data,
                                     int body_id, const double p_world[3],
                                     double s_world[3]) const;

 private:
  Terrain terrain_;

  // Visualization IDs for norm-clearance preview (head + 4 knees)
  int head_site_id_vis_ = -1;
  int knee_body_id_[4] = {-1, -1, -1, -1};  // order: FL, FR, HL, HR

    // Mocap ids for support boxes (order: FL, FR, HL, HR)
    int box_mocap_id_[4] = {-1, -1, -1, -1};
    // Foot geom ids for convenience (order: FL, FR, HL, HR)
    int foot_geom_id_boxref_[4] = {-1, -1, -1, -1};
    // Box half-height in meters (size[2]) to position center below surface
    double box_half_height_ = 0.04;

    // Generic mapping: for any named robot collision geom G, if a mocap body
    // named "box_"+G exists, we will update its pose each step.
    struct PairMapEntry {
      int geom_id = -1;
      int mocap_id = -1;
      double half_h = 0.02;
    };
    std::vector<PairMapEntry> generic_pairs_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_QUADRUPED_H_
