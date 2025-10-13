#include <iostream>
#include <chrono>
#include <limits>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <thread>
#include <algorithm>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/algorithm.hpp>
#include <pagmo/problem.hpp>
#include <fstream>
#include <sstream>
#include <random>

#include "mjpc/weight_opt/pagmo_mjtwin_problem.h"
#include "mjpc/weight_opt/weights_opt.h"
#include "mjpc/utilities.h"



ABSL_FLAG(double, episode_time, 10.0, "episode rollout seconds");
ABSL_FLAG(int,    pop_size, 40, "population size");
ABSL_FLAG(int,    iters, 120, "evolution iterations");
ABSL_FLAG(double, success_radius_m, 0.10, "success radius in meters");
ABSL_FLAG(int,    log_every, 1, "print progress every N iterations (0=off)");
ABSL_FLAG(int,    save_video_every, 4, "save a short video every N iterations (0=off)");
ABSL_FLAG(int,    video_width, 960, "video width");
ABSL_FLAG(int,    video_height, 540, "video height");
ABSL_FLAG(double, video_fps, 24.0, "video fps");
ABSL_FLAG(double, video_duration, 6.0, "video duration seconds per snapshot");
ABSL_FLAG(bool,   save_final_video, true, "save final video at the end");
ABSL_FLAG(bool,   final_video_fresh, true, "render a fresh final video (skip copying existing eval video)");
ABSL_FLAG(std::string, out_dir, "exps", "output directory for csv, weights, videos (relative paths are under build dir)");
ABSL_FLAG(std::string, video_basename, "mjtwin_run", "basename for output video files");
ABSL_FLAG(bool,   pace_realtime, true, "pace physics to wall-clock time (sleep to sync); false runs as fast as possible");
ABSL_FLAG(bool,   elitism, true, "inject best-so-far into population each iteration");
ABSL_FLAG(bool,   save_iter_champion_video, false, "save per-iteration champion video (disabled by default)");



int main(int argc, char** argv) {

  absl::ParseCommandLine(argc, argv);
  int planner_threads = std::max(1, mjpc::NumAvailableHardwareThreads() - 3);
  double episode_time = absl::GetFlag(FLAGS_episode_time);
  int pop_size        = absl::GetFlag(FLAGS_pop_size);
  int iters           = absl::GetFlag(FLAGS_iters);

  std::cout << "Differential Evolution" << std::endl;
  std::cout << "Configuration:" << std::endl
            << "   episode_time       = " << episode_time << std::endl
            << "   population_size    = " << pop_size << std::endl
            << "   optimization iters = " << iters << std::endl;

  std::vector<std::string> names = mjpc::weights_opt::ListCostTermNames("mjTwin");
  std::vector<char> mask(names.size(), 1);

  for (size_t i = 0; i < names.size(); ++i) {
    std::string n = names[i];
    for (auto& c : n) c = std::tolower(c);
    if (n == "orientation" || n == "angmom" || n == "height") mask[i] = 0;
    if (n == "footcost" || n == "effort") mask[i] = 0;
    if (n == "normclear") mask[i] = 0;
  }

  std::vector<std::string> opt_names;
  for (size_t i = 0; i < names.size(); ++i)
    if (mask[i])
      opt_names.push_back(names[i]);

  mjpc::weights_opt::Runner init_runner("mjTwin", planner_threads, episode_time);
  std::vector<double> init_defaults = init_runner.DefaultWeights();
  std::vector<double> init_opt_values;

  for (size_t i = 0; i < names.size(); ++i)
    if (i < mask.size() && mask[i])
      init_opt_values.push_back(i < init_defaults.size() ? init_defaults[i] : 0.0);
  
  std::cout << "Optimizing: " << opt_names.size() << " out of " << names.size() << " terms" << std::endl;
  std::cout << "Default Weights: ";
  for (size_t i = 0; i < opt_names.size() && i < init_opt_values.size(); ++i) {
    if (i > 0) std::cout << " - ";
    std::cout << opt_names[i] << " = " << init_opt_values[i];
  }
  std::cout << std::endl;


  pagmo::problem prob{mjpc::weights_opt::MjTwinWeightsProblem(names, mask, planner_threads, episode_time, absl::GetFlag(FLAGS_success_radius_m))};
  pagmo::algorithm algo{pagmo::de{}};
  auto bounds = prob.get_bounds();
  auto lb = bounds.first;
  auto ub = bounds.second;
  auto nx = prob.get_nx();
  pagmo::population pop{prob, 0u};
  // RNG for population init
  std::mt19937 rng(std::random_device{}());
  // single-line progress for initial population generation
  auto print_init_progress = [&](int done, int total) {
    std::cout << "\rInitializing population " << done << "/" << total << std::flush;
  };
  int init_done = 0;
  print_init_progress(init_done, pop_size);
  if (absl::GetFlag(FLAGS_elitism)) {
    // push default weights first (counts as one evaluation)
    pop.push_back(init_opt_values);
    init_done = 1;
    print_init_progress(init_done, pop_size);
    for (int k = 1; k < pop_size; ++k) {
      pagmo::vector_double x(nx, 0.0);
      for (std::size_t d = 0; d < nx; ++d) {
        std::uniform_real_distribution<double> dist(lb[d], ub[d]);
        x[d] = dist(rng);
      }
      pop.push_back(x);
      init_done++;
      print_init_progress(init_done, pop_size);
    }
  } else {
    // fully random initial population
    for (int k = 0; k < pop_size; ++k) {
      pagmo::vector_double x(nx, 0.0);
      for (std::size_t d = 0; d < nx; ++d) {
        std::uniform_real_distribution<double> dist(lb[d], ub[d]);
        x[d] = dist(rng);
      }
      pop.push_back(x);
      init_done++;
      print_init_progress(init_done, pop_size);
    }
  }
  std::cout << std::endl;

  // Prepare initial-evals routing; initial population evaluated with iter=-1.
  mjpc::weights_opt::SetCurrentIterationIndex(-1);
  std::string out_dir = absl::GetFlag(FLAGS_out_dir);
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  std::ofstream hist(out_dir + "/weights_over_time.csv");
  hist << "iter";
  for (const auto& n : opt_names) hist << "," << n;
  hist << "\n";
  hist.flush();
  int vW = absl::GetFlag(FLAGS_video_width);
  int vH = absl::GetFlag(FLAGS_video_height);
  double vFPS = absl::GetFlag(FLAGS_video_fps);
  double vDur = std::min(absl::GetFlag(FLAGS_video_duration), episode_time);
  bool pace_realtime = absl::GetFlag(FLAGS_pace_realtime);
  auto can_render = [](){
    const char* gl = std::getenv("MUJOCO_GL");
    std::string v = gl ? std::string(gl) : std::string("");
    for (auto& c : v) c = std::tolower(c);
    if (v == "osmesa") return true;
    if (v == "glfw") return std::getenv("DISPLAY") && *std::getenv("DISPLAY");
    return false;
  };
  bool render_ok = can_render();
  if (!render_ok) {
    const char* gl = std::getenv("MUJOCO_GL");
    std::string backend = gl ? std::string(gl) : std::string("");
    std::cout << "video disabled for backend='" << backend << "' (supported: osmesa, or glfw with DISPLAY)." << std::endl;
  }

  // Clear any seed-eval stats so iter_0 counts only its own evaluations
  {
    int _t = 0, _f = 0;
    mjpc::weights_opt::GetAndResetEvaluationStats(&_t, &_f);
  }
  // Start optimization iterations; reset FitEval counter and set iter to 0
  mjpc::weights_opt::ResetFitEvalCounter();
  pagmo::vector_double prev_best = init_opt_values;
  pagmo::vector_double last_iter_best;
  double last_iter_best_f = std::numeric_limits<double>::infinity();
  bool have_prev_best = absl::GetFlag(FLAGS_elitism);
  for (int i = 0; i < iters; ++i) {
    std::cout << "----- Iteration " << i << " -----" << std::endl;
    mjpc::weights_opt::SetCurrentIterationIndex(i);
    // Inject elite (best-so-far) into population before evolving
    if (absl::GetFlag(FLAGS_elitism) && have_prev_best) {
      if (pop.size() > 0 && prev_best.size() == pop.get_x()[0].size()) {
        pop.set_x(0, prev_best);
      }
    }
    pop = algo.evolve(pop);
    int iter_evals = 0, iter_fell = 0;
    mjpc::weights_opt::GetAndResetEvaluationStats(&iter_evals, &iter_fell);
    if (iter_evals > pop_size) iter_evals = pop_size;

    // Determine champion strictly from evaluations recorded in this iteration
    // to ensure consistency between printed costs and FitEval logs.
    std::vector<mjpc::weights_opt::EvalInfo> eval_infos;
    bool have_infos = mjpc::weights_opt::GetIterationEvalInfos(i, /*max_count=*/0, &eval_infos);
    pagmo::vector_double best;
    double best_f = std::numeric_limits<double>::infinity();
    int champ_eval_id = -1;
    if (have_infos && !eval_infos.empty()) {
      // Pick argmin cost among this iteration's evaluations
      size_t best_idx = 0;
      for (size_t ei = 0; ei < eval_infos.size(); ++ei) {
        if (ei == 0 || eval_infos[ei].cost < eval_infos[best_idx].cost) best_idx = ei;
      }
      champ_eval_id = eval_infos[best_idx].eval_id;
      best_f = eval_infos[best_idx].cost;
      best = eval_infos[best_idx].optimized_x;
    } else {
      // Fallback: use population champion if no evaluation infos were recorded
      auto f = pop.champion_f();
      if (!f.empty()) { best_f = f[0]; best = pop.champion_x(); }
      if (!best.empty()) {
        mjpc::weights_opt::FindEvalIdForOptimizedWeights(i, best, &champ_eval_id);
      }
    }

    std::cout << "Iteration " << i << ", cost = " << (std::isfinite(best_f) ? best_f : 0.0);
    if (champ_eval_id >= 0) std::cout << ", champ = " << champ_eval_id;
    std::cout << std::endl;
    std::ostringstream line;
    line << i;
    for (size_t k = 0; k < best.size(); ++k) line << "," << best[k];
    line << "\n";
    hist << line.str();
    hist.flush();
    // no extra per-iteration cost line
    // Optionally save per-iteration champion video if enabled.
    if (absl::GetFlag(FLAGS_save_iter_champion_video) && render_ok && std::isfinite(best_f) && best.size() == opt_names.size()) {
      std::string iter_dir = out_dir + std::string("/iter_") + std::to_string(i);
      std::error_code ec_iter;
      std::filesystem::create_directories(iter_dir, ec_iter);
      mjpc::weights_opt::Runner runner("mjTwin", planner_threads, episode_time);
      std::string out_video;
      std::string base = absl::GetFlag(FLAGS_video_basename) + std::string("_iter_") + std::to_string(i) + std::string("_champ");
      if (champ_eval_id >= 0) base += std::string("_eval_") + std::to_string(champ_eval_id);
      bool copied = false;
      // If we know the exact eval id, prefer copying the per-evaluation video to ensure identical behavior
      if (champ_eval_id >= 0) {
        int src_iter = -1;
        std::vector<double> _ox;
        std::string src_path;
        if (mjpc::weights_opt::GetEvalMetadata(champ_eval_id, &src_iter, &_ox, &src_path, nullptr)) {
          if (!src_path.empty() && std::filesystem::exists(src_path)) {
            std::string dst_path = iter_dir + std::string("/") + base + std::string(".mp4");
            std::error_code ecc;
            if (std::filesystem::copy_file(src_path, dst_path, std::filesystem::copy_options::overwrite_existing, ecc)) {
              copied = true;
              out_video = dst_path;
              std::cout << "Champion video saved at: " << out_video << " (copied from eval_" << champ_eval_id << ")" << std::endl;
            }
          }
        }
      }
      if (!copied) {
        // reconstruct full weight vector from defaults + mask using champion (optimized) weights
        std::vector<double> dflt_full = runner.DefaultWeights();
        std::vector<double> full = dflt_full;
        for (size_t idx = 0, j = 0; idx < names.size() && j < best.size(); ++idx) {
          if (idx < mask.size() && mask[idx]) { if (idx < full.size()) full[idx] = best[j++]; }
        }
        runner.RenderVideo(full, iter_dir, vW, vH, vFPS, vDur, &out_video, base, /*pace_realtime=*/pace_realtime);
        if (!out_video.empty()) std::cout << "Champion video saved at: " << out_video << std::endl;
        else std::cout << "Iteration " << i << ": champion video skipped (rendering failed or disabled)." << std::endl;
      }
    }

    // update best-so-far for next iteration elitism
    if (!best.empty()) { prev_best = best; have_prev_best = true; }
    last_iter_best = best; last_iter_best_f = best_f;
  }
  hist.close();
  pagmo::vector_double champion = pop.champion_x();
  pagmo::vector_double champion_f = pop.champion_f();

  std::cout << "Final optimized weights:" << std::endl;
  for (size_t i = 0; i < opt_names.size() && i < champion.size(); ++i) {
    std::cout << opt_names[i] << ": " << champion[i] << std::endl;
  }
  std::ofstream ofs(out_dir + "/mjtwin_best_weights.txt");
  for (size_t i = 0; i < opt_names.size() && i < champion.size(); ++i) {
    ofs << opt_names[i] << "," << champion[i] << "\n";
  }
  ofs << "cost," << (champion_f.empty() ? 0.0 : champion_f[0]) << "\n";
  ofs.close();

  std::cout << "Artifacts saved in directory '" << out_dir << "': weights_over_time.csv, mjtwin_best_weights.txt" << std::endl;
  if (absl::GetFlag(FLAGS_save_final_video) && render_ok) {
    const pagmo::vector_double& final_w = champion;
    std::string final_src;
    int final_eval_id = -1;
    // Optionally try to copy exact eval video if fresh rendering is disabled
    if (!absl::GetFlag(FLAGS_final_video_fresh)) {
      if (mjpc::weights_opt::FindEvalIdForOptimizedWeightsAtOrBefore(iters - 1, final_w, &final_eval_id)) {
        int src_iter = -1; std::vector<double> _ox; std::string src_path;
        if (mjpc::weights_opt::GetEvalMetadata(final_eval_id, &src_iter, &_ox, &src_path, nullptr) && src_iter >= 0) {
          std::string src_iter_dir = out_dir + std::string("/iter_") + std::to_string(src_iter);
          std::string src = src_iter_dir + "/" + absl::GetFlag(FLAGS_video_basename) + std::string("_eval_") + std::to_string(final_eval_id) + ".mp4";
          std::string dst = out_dir + std::string("/") + absl::GetFlag(FLAGS_video_basename) + std::string(".mp4");
          std::error_code ecc;
          if (std::filesystem::exists(src) && std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, ecc)) {
            std::cout << "Final video saved at: " << dst << " (copied from eval_" << final_eval_id << ")" << std::endl;
            return 0;
          }
        }
      }
    }
    std::cout << "Rendering final video (global best weights)" << std::endl;
    mjpc::weights_opt::Runner runner("mjTwin", planner_threads, episode_time);
    std::string out_video;
    // reconstruct full vector for final render
    std::vector<double> dflt_full = runner.DefaultWeights();
    std::vector<double> full = dflt_full;
    for (size_t idx = 0, j = 0; idx < names.size() && j < final_w.size(); ++idx) {
      if (idx < mask.size() && mask[idx]) { if (idx < full.size()) full[idx] = final_w[j++]; }
    }
    bool ok = runner.RenderVideo(
        full,
        out_dir,
        absl::GetFlag(FLAGS_video_width),
        absl::GetFlag(FLAGS_video_height),
        absl::GetFlag(FLAGS_video_fps),
        absl::GetFlag(FLAGS_video_duration),
        &out_video,
        absl::GetFlag(FLAGS_video_basename),
        /*pace_realtime=*/pace_realtime);
    if (ok) std::cout << "Final video saved at: " << out_video << std::endl;
    else std::cout << "Final video skipped (rendering failed or disabled)." << std::endl;
  } else if (absl::GetFlag(FLAGS_save_final_video)) {
    std::cout << "Final video skipped (rendering not available). For headless rendering set MUJOCO_GL=osmesa or use a GUI-backed glfw display." << std::endl;
  }
  return 0;
}


