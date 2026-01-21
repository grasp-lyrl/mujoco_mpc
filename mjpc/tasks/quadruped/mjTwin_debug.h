#ifndef MJPC_TASKS_QUADRUPED_MJTWIN_DEBUG_H_
#define MJPC_TASKS_QUADRUPED_MJTWIN_DEBUG_H_

#include <memory>
#include <string>

#include "mjpc/tasks/quadruped/mjTwin.h"

namespace mjpc {

class MjTwinDebug : public MjTwin {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  void ResetLocked(const mjModel* model) override;
  void TransitionLocked(mjModel* model, mjData* data) override;
  void TransitionEnvOnlyLocked(mjModel* model, mjData* data) override;
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;

  class ResidualFn : public Quadruped::ResidualFn {
   public:
    explicit ResidualFn(const MjTwinDebug* task) : Quadruped::ResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    void CopyFrom(const Quadruped::ResidualFn& src, const MjTwinDebug* task) {
      static_cast<Quadruped::ResidualFn&>(*this) = src;
      // Re-bind to this task instance after the base copy.
      this->task_ = task;
    }
  };

  MjTwinDebug() : debug_residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(debug_residual_);
  }
  ResidualFn* InternalResidual() override { return &debug_residual_; }

 private:
  friend class ResidualFn;
  ResidualFn debug_residual_;
  bool initialized_ = false;
  double initial_foot_pos_[Quadruped::ResidualFn::kNumFoot][3] = {};
  double initial_torso_pos_[3] = {0.0, 0.0, 0.0};
  double fr_bezier_ctrl_[4][3] = {};
};

}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_MJTWIN_DEBUG_H_
