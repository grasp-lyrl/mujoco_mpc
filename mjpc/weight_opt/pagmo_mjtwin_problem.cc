#include "mjpc/weight_opt/pagmo_mjtwin_problem.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <random>

#include <absl/flags/declare.h>
#include <absl/flags/flag.h>

#include "mjpc/weight_opt/weights_opt.h"

ABSL_DECLARE_FLAG(std::string, out_dir);
ABSL_DECLARE_FLAG(int, video_width);
ABSL_DECLARE_FLAG(int, video_height);
ABSL_DECLARE_FLAG(double, video_fps);
ABSL_DECLARE_FLAG(double, video_duration);
ABSL_DECLARE_FLAG(bool, pace_realtime);
ABSL_DECLARE_FLAG(std::string, video_basename);
ABSL_DECLARE_FLAG(bool, save_eval_videos);
ABSL_DECLARE_FLAG(int, save_eval_video_every);
ABSL_DECLARE_FLAG(bool, save_initial_eval_videos);

namespace mjpc {
namespace weights_opt {

pagmo::vector_double MjTwinWeightsProblem::fitness(const pagmo::vector_double& x) const {
  mjpc::weights_opt::Runner runner("mjTwin", planner_threads_, total_time_);
  pagmo::vector_double defaults = runner.DefaultWeights();
  std::vector<std::pair<double,double>> bounds = runner.WeightBounds();
  // working copy of optimized weights; may be resampled on early fall
  pagmo::vector_double cur_x = x;
  // rng for resampling
  thread_local std::mt19937 rng{std::random_device{}()};
  int attempts = 0;
  const int kMaxAttempts = 50;
  double cost = 0.0;
  bool fell = false;
  double fall_t = 0.0;
  double fall_dx = 0.0;
  while (true) {
    pagmo::vector_double full = defaults;
    for (size_t i = 0, j = 0; i < names_.size() && j < cur_x.size(); ++i) {
      if (i < mask_.size() && mask_[i]) { full[i] = cur_x[j++]; }
    }
    // evaluate with real-time pacing; disable verbose per-second prints
    cost = runner.Evaluate(full, success_radius_m_, false, /*pace_realtime=*/true);
    fell = GetAndClearLastFallInfo(&fall_t, &fall_dx);
    if (fell && fall_dx <= 0.8 && attempts < kMaxAttempts) {
      // discard and resample a new cur_x within bounds for masked dims
      attempts++;
      for (size_t i = 0, j = 0; i < names_.size() && j < cur_x.size(); ++i) {
        if (i < mask_.size() && mask_[i]) {
          double lo = (i < bounds.size()) ? bounds[i].first : 0.0;
          double hi = (i < bounds.size()) ? bounds[i].second : 1.0;
          std::uniform_real_distribution<double> dist(lo, hi);
          cur_x[j++] = dist(rng);
        }
      }
      continue;
    }
    break;
  }
  static std::atomic<int> eval_counter{0};
  int iter_idx_now = mjpc::weights_opt::GetCurrentIterationIndex();
  bool is_initial_seed = (iter_idx_now < 0);
  int eval_id = is_initial_seed ? (1 + eval_counter.fetch_add(1))
                                : mjpc::weights_opt::NextFitEvalId();
  // Optionally render per-evaluation video into appropriate subfolder
  bool should_save = absl::GetFlag(FLAGS_save_eval_videos);
  int every = absl::GetFlag(FLAGS_save_eval_video_every);
  if (every <= 0) should_save = false;
  // allow toggling initial seed video generation separately
  if (is_initial_seed && !absl::GetFlag(FLAGS_save_initial_eval_videos)) should_save = false;
  if (should_save && ((eval_id % every) == 0)) {
    std::string out_root = absl::GetFlag(FLAGS_out_dir);
    if (out_root.empty()) out_root = "exps";
    std::string out_dir;
    if (is_initial_seed) out_dir = out_root + "/initial_evals";
    else if (iter_idx_now >= 0) out_dir = out_root + "/iter_" + std::to_string(iter_idx_now);
    else out_dir = out_root;  // fallback
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    int vW = absl::GetFlag(FLAGS_video_width);
    int vH = absl::GetFlag(FLAGS_video_height);
    double vFPS = absl::GetFlag(FLAGS_video_fps);
    double vDur = std::min(absl::GetFlag(FLAGS_video_duration), total_time_);
    bool pace = absl::GetFlag(FLAGS_pace_realtime);
    std::string base = absl::GetFlag(FLAGS_video_basename) + std::string("_eval_") + std::to_string(eval_id);
    std::string out_video;
    // reconstruct full from cur_x used for final evaluation
    pagmo::vector_double full_for_video = defaults;
    for (size_t i = 0, j = 0; i < names_.size() && j < cur_x.size(); ++i) {
      if (i < mask_.size() && mask_[i]) { full_for_video[i] = cur_x[j++]; }
    }
    runner.RenderVideo(full_for_video, out_dir, vW, vH, vFPS, vDur, &out_video, base, /*pace_realtime=*/pace);
    mjpc::weights_opt::RecordEvaluationMetadata(eval_id, is_initial_seed ? -1 : iter_idx_now, cur_x, full_for_video, fell, cost, out_video);
  } else {
    // still record metadata but without a video path
    std::string empty;
    // reconstruct full from cur_x used for final evaluation
    pagmo::vector_double full_final = defaults;
    for (size_t i = 0, j = 0; i < names_.size() && j < cur_x.size(); ++i) {
      if (i < mask_.size() && mask_[i]) { full_final[i] = cur_x[j++]; }
    }
    mjpc::weights_opt::RecordEvaluationMetadata(eval_id, is_initial_seed ? -1 : iter_idx_now, cur_x, full_final, fell, cost, empty);
  }
  // concise one-line output (skip numbering/printing for initial seed)
  if (!is_initial_seed) {
  std::cout << "FitEval #" << eval_id << ": cost=" << cost << ", weights=[";
    for (size_t wi = 0; wi < cur_x.size(); ++wi) { if (wi) std::cout << ", "; std::cout << cur_x[wi]; }
    std::cout << "]";
    if (fell) {
      std::cout << ", fell at t=" << fall_t << "s";
      if (fall_dx > 0.0) std::cout << ", dx=" << fall_dx << "m";
    }
    // For kept trajectories (no final fall), append number of rejected resamples
    if (!fell && attempts > 0) {
      std::cout << ", rejected=" << attempts;
    }
    std::cout << std::endl;
  }
  return {cost};
}

std::pair<pagmo::vector_double, pagmo::vector_double> MjTwinWeightsProblem::get_bounds() const {
  mjpc::weights_opt::Runner runner("mjTwin", planner_threads_, total_time_);
  pagmo::vector_double dflt = runner.DefaultWeights();
  std::vector<std::pair<double,double>> bounds = runner.WeightBounds();
  pagmo::vector_double lb, ub;
  for (size_t i = 0; i < names_.size() && i < dflt.size() && i < bounds.size(); ++i) {
    if (i < mask_.size() && mask_[i]) {
      lb.push_back(bounds[i].first);
      ub.push_back(bounds[i].second);
    }
  }
  return {lb, ub};
}

}  // namespace weights_opt
}  // namespace mjpc


