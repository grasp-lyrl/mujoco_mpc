#ifndef MJPC_TASKS_QUADRUPED_QUADRUPED_H_
#define MJPC_TASKS_QUADRUPED_QUADRUPED_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/quadruped/terrain.h"

namespace mjpc {

class Quadruped : public Task {
  public:
    std::string Name() const override = 0;
    std::string XmlPath() const override = 0;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Quadruped* task) : mjpc::BaseResidualFn(task) {}
    ResidualFn(const ResidualFn&) = default;
    void Residual(const mjModel* model, const mjData* data, double* residual) const override;

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
      {1,          1,       0,         0,        1,        1},      // stand
      {0.75,       1,       0.03,      0.2,      1,        1},      // walk
      {0.8,        1,       0.03,      0.2,      1,        1},      // trot
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
    // constexpr static double kHeightBiped = 0.50;       // meter
    constexpr static double kHeightBiped = 0.60;       // meter

    // radius of foot geoms
    // constexpr static double kFootRadius = 0.02;       // meter
    constexpr static double kFootRadius = 0.022;       // meter

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


    // ============= cost terms =============

    int UprightCost(const mjData* data, double* residual,
                    int counter) const;
    int HeightCost(const mjData* data, const double torso_pos[3],
                   double height_goal, const double avg_foot_pos[3],
                   double* residual, int counter) const;
    int PositionCost(const mjData* data, double* goal_pos,
                     double* residual, int counter) const;
    int GaitCost(const mjModel* model, const mjData* data,
                 const double torso_pos[3], bool is_biped,
                 double* foot_pos[kNumFoot], double* goal_pos,
                 double* residual, int counter) const;
    int BalanceCost(const mjModel* model, const mjData* data,
                    double height_goal, const double avg_foot_pos[3],
                    double* compos, double* residual,
                    int counter) const;
    int EffortCost(const mjModel* model, const mjData* data,
                   double* residual, int counter) const;
    int PostureCost(const mjModel* model, const mjData* data,
                    double* residual, int counter) const;
    int YawCost(const mjModel* model, const mjData* data,
                const double* torso_xmat, double* residual,
                int counter) const;
    int AngularMomentumCost(const mjModel* model, const mjData* data,
                            double* residual, int counter) const;
    int ClearanceCost(const mjModel* model, const mjData* data,
                      double* residual, int counter) const;

   public:
    void GetProjectedFoothold(const mjData* data, const double query[3],
                              double safe_xy[2]) const;
   private:
    int CostFlatGround(const mjModel* model, const mjData* data, A1Foot foot,
                       const double foot_pos[3], const double query[3],
                       double step_amplitude, double* residual,
                       int counter) const;
    int CostRoughGround(const mjModel* model, const mjData* data, A1Foot foot,
                        const double foot_pos[3], const double query[3],
                        double step_amplitude, double current_step_height,
                        double* residual, int counter) const;

   public:
    //  ============  task state variables, managed by Transition  ============
    A1Mode current_mode_         = kModeQuadruped;
    double last_transition_time_ = -1;

    // common mode states
    double mode_start_time_   = 0;
    double position_[3]       = {0};

    // walk states
    double heading_[2]        = {0};
    double speed_             = 0;
    double angvel_            = 0;

    // gait-related states
    double current_gait_      = kGaitWalk;
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
    int terrain_type_param_id_ = -1;
    int upright_cost_id_      = -1;
    int balance_cost_id_      = -1;
    int height_cost_id_       = -1;
    
    int foot_geom_id_[kNumFoot];

    int shoulder_body_id_[kNumFoot];
    int shoulder_box_mocap_id_[kNumFoot];

    int knee_body_id_[kNumFoot];
    int knee_box_mocap_id_[kNumFoot];

    int lidar_geom_id_ = -1;
    int lidar_box_mocap_id_ = -1;
    double box_half_size_[3] = {0.0, 0.0, 0.0};
    
    const Terrain* terrain_ = nullptr;
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

  friend class ResidualFn;
  ResidualFn residual_;
};


}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_QUADRUPED_H_
