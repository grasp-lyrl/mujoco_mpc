#ifndef MJPC_WEIGHT_OPT_PAGMO_MJTWIN_PROBLEM_H_
#define MJPC_WEIGHT_OPT_PAGMO_MJTWIN_PROBLEM_H_

#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <pagmo/types.hpp>

#include "mjpc/weight_opt/weights_opt.h"

namespace mjpc {
namespace weights_opt {

class MjTwinWeightsProblem {
 public:
  MjTwinWeightsProblem() = default;
  MjTwinWeightsProblem(std::vector<std::string> names,
                       std::vector<char> optimize_mask,
                       int planner_threads,
                       double total_time,
                       double success_radius_m)
      : names_(std::move(names)),
        mask_(std::move(optimize_mask)),
        planner_threads_(planner_threads),
        total_time_(total_time),
        success_radius_m_(success_radius_m) {}

  std::vector<std::string> get_names() const { return names_; }
  pagmo::vector_double fitness(const pagmo::vector_double& x) const;
  std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const;
  bool has_gradient() const { return false; }
  std::string get_name() const { return "MjTwinWeightsProblem"; }
  std::size_t get_nobj() const { return 1u; }
  std::size_t get_nec() const { return 0u; }
  std::size_t get_nic() const { return 0u; }
  std::size_t get_nix() const { return 0u; }
  std::size_t get_nx() const {
    std::size_t n = 0;
    for (auto c : mask_) if (c) ++n;
    return n;
  }

 private:
  std::vector<std::string> names_;
  std::vector<char> mask_;
  int planner_threads_;
  double total_time_;
  double success_radius_m_;
};

}  // namespace weights_opt
}  // namespace mjpc

#endif  // MJPC_WEIGHT_OPT_PAGMO_MJTWIN_PROBLEM_H_


