#ifndef MJPC_WEIGHT_OPT_WEIGHTS_OPT_H_
#define MJPC_WEIGHT_OPT_WEIGHTS_OPT_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/threadpool.h"

namespace mjpc {
namespace weights_opt {

class Runner {
 public:
  Runner(const std::string& task_name,
         int planner_thread_count,
         double time_limit_seconds);
  ~Runner();

  std::vector<double> DefaultWeights() const;
  std::vector<std::pair<double,double>> WeightBounds() const;
  double Evaluate(const std::vector<double>& weights,
                  double success_radius_m,
                  bool verbose_progress,
                  bool pace_realtime = true);

  bool RenderVideo(const std::vector<double>& weights,
                   const std::string& out_dir,
                   int width,
                   int height,
                   double fps,
                   double duration_seconds,
                   std::string* out_video_path,
                   const std::string& out_basename = "mjtwin_run",
                   bool pace_realtime = true);

 private:
  std::string task_;
  int planner_threads_;
  double time_limit_;
  std::vector<std::string> names_;
  std::unique_ptr<mjpc::Agent> agent_;
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  std::unique_ptr<ThreadPool> pool_;
};

// Optimization-iteration context & metadata utilities
void SetCurrentIterationIndex(int iteration_index);
int GetCurrentIterationIndex();
// reset FitEval numbering (used after initial population evals)
void ResetFitEvalCounter();
// fetch next FitEval id (1-based), used only for optimization (not seeds)
int NextFitEvalId();

// Record per-evaluation metadata (weights, outcome, paths) for later queries.
void RecordEvaluationMetadata(
    int eval_id,
    int iteration_index,
    const std::vector<double>& optimized_weights_x,
    const std::vector<double>& full_weights,
    bool fell,
    double cost,
    const std::string& video_path);

// Find the eval id for a given optimized-weights vector within an iteration.
// Returns true when a match is found (exact match on vector size and elements).
bool FindEvalIdForOptimizedWeights(
    int iteration_index,
    const std::vector<double>& optimized_weights_x,
    int* out_eval_id);

// Find the most recent eval id at or before max_iteration matching weights.
bool FindEvalIdForOptimizedWeightsAtOrBefore(
    int max_iteration_index,
    const std::vector<double>& optimized_weights_x,
    int* out_eval_id);

// Retrieve metadata for a given eval id. Returns true on success.
bool GetEvalMetadata(
    int eval_id,
    int* out_iteration_index,
    std::vector<double>* out_optimized_x,
    std::string* out_video_path,
    double* out_cost);

struct EvalInfo {
  int eval_id;
  double cost;
  std::vector<double> optimized_x;
};

// Retrieve up to max_count evaluation infos for a given iteration, in
// chronological order of occurrence. Returns true if any were found.
bool GetIterationEvalInfos(
    int iteration_index,
    int max_count,
    std::vector<EvalInfo>* out_infos);

// Returns true if the last call to Runner::Evaluate() terminated early due to
// head-ground contact. If true, writes the fall time (seconds since episode
// start) to fall_time_out and clears the flag.
bool GetAndClearLastFall(double* fall_time_out);

// Returns true if the last call to Runner::Evaluate() terminated early due to
// head-ground contact and provides both fall time (since episode start) and
// the absolute x-axis displacement of the trunk from its initial position at
// the moment of the fall. If true, writes outputs and clears the flag/state.
bool GetAndClearLastFallInfo(double* fall_time_out, double* fall_dx_out);

// Record evaluation outcome for aggregated stats.
void RecordEvaluation(bool fell);

// Retrieve and reset aggregated evaluation stats for the last batch.
// Returns true if stats were available; writes totals to out params.
bool GetAndResetEvaluationStats(int* total_evals, int* fell_evals);

double EvaluateCostForWeights(
    const std::string& task_name,
    const std::vector<std::pair<std::string, double>>& name_value_weights,
    int planner_thread_count,
    double total_time_seconds,
    bool verbose_progress = true);

std::vector<std::string> ListCostTermNames(const std::string& task_name);

std::vector<std::pair<std::string,double>> DefaultCostWeights(
    const std::string& task_name);

}  // namespace weights_opt
}  // namespace mjpc

#endif  // MJPC_WEIGHT_OPT_WEIGHTS_OPT_H_


