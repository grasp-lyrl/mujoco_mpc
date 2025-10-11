#include "mjpc/weight_opt/weights_opt.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <sys/wait.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include "mjpc/agent.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

namespace mjpc {
namespace weights_opt {

namespace { Task* g_task_for_callback; }
namespace {
std::atomic<bool> g_last_fall{false};
double g_last_fall_time{0.0};
double g_last_fall_dx{0.0};
std::atomic<int> g_eval_total{0};
std::atomic<int> g_eval_fell{0};
// iteration context and metadata storage
std::atomic<int> g_current_iteration{-1};
static std::atomic<int> g_fiteval_counter{0};
struct EvalMeta {
  int eval_id;
  int iteration_index;
  std::vector<double> optimized_x;
  std::vector<double> full_w;
  bool fell;
  double cost;
  std::string video_path;
};
static std::mutex g_meta_mu;
static std::vector<EvalMeta> g_eval_meta;
}

bool GetAndClearLastFall(double* fall_time_out) {
  if (g_last_fall.exchange(false)) {
    if (fall_time_out) *fall_time_out = g_last_fall_time;
    g_last_fall_time = 0.0;
    g_last_fall_dx = 0.0;
    return true;
  }
  return false;
}

bool GetAndClearLastFallInfo(double* fall_time_out, double* fall_dx_out) {
  if (g_last_fall.exchange(false)) {
    if (fall_time_out) *fall_time_out = g_last_fall_time;
    if (fall_dx_out) *fall_dx_out = g_last_fall_dx;
    g_last_fall_time = 0.0;
    g_last_fall_dx = 0.0;
    return true;
  }
  return false;
}

void RecordEvaluation(bool fell) {
  g_eval_total.fetch_add(1);
  if (fell) g_eval_fell.fetch_add(1);
}

bool GetAndResetEvaluationStats(int* total_evals, int* fell_evals) {
  int t = g_eval_total.exchange(0);
  int f = g_eval_fell.exchange(0);
  if (t == 0) return false;
  if (total_evals) *total_evals = t;
  if (fell_evals) *fell_evals = f;
  return true;
}

void SetCurrentIterationIndex(int iteration_index) { g_current_iteration.store(iteration_index); }
int GetCurrentIterationIndex() { return g_current_iteration.load(); }
void ResetFitEvalCounter() { g_fiteval_counter.store(0); }
int NextFitEvalId() { return 1 + g_fiteval_counter.fetch_add(1); }

void RecordEvaluationMetadata(
    int eval_id,
    int iteration_index,
    const std::vector<double>& optimized_weights_x,
    const std::vector<double>& full_weights,
    bool fell,
    double cost,
    const std::string& video_path) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  g_eval_meta.push_back(EvalMeta{eval_id, iteration_index, optimized_weights_x, full_weights, fell, cost, video_path});
}

bool FindEvalIdForOptimizedWeights(
    int iteration_index,
    const std::vector<double>& optimized_weights_x,
    int* out_eval_id) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  // scan from newest to oldest to prefer most recent matching eval
  for (auto it = g_eval_meta.rbegin(); it != g_eval_meta.rend(); ++it) {
    const auto& m = *it;
    if (m.iteration_index != iteration_index) continue;
    if (m.optimized_x.size() != optimized_weights_x.size()) continue;
    bool same = true;
    for (size_t i = 0; i < m.optimized_x.size(); ++i) {
      if (m.optimized_x[i] != optimized_weights_x[i]) { same = false; break; }
    }
    if (same) { if (out_eval_id) *out_eval_id = m.eval_id; return true; }
  }
  return false;
}

bool FindEvalIdForOptimizedWeightsAtOrBefore(
    int max_iteration_index,
    const std::vector<double>& optimized_weights_x,
    int* out_eval_id) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  for (auto it = g_eval_meta.rbegin(); it != g_eval_meta.rend(); ++it) {
    const auto& m = *it;
    if (m.iteration_index > max_iteration_index) continue;
    if (m.optimized_x.size() != optimized_weights_x.size()) continue;
    bool same = true;
    for (size_t i = 0; i < m.optimized_x.size(); ++i) {
      if (m.optimized_x[i] != optimized_weights_x[i]) { same = false; break; }
    }
    if (same) { if (out_eval_id) *out_eval_id = m.eval_id; return true; }
  }
  return false;
}

bool GetEvalMetadata(
    int eval_id,
    int* out_iteration_index,
    std::vector<double>* out_optimized_x,
    std::string* out_video_path,
    double* out_cost) {
  std::lock_guard<std::mutex> lk(g_meta_mu);
  for (const auto& m : g_eval_meta) {
    if (m.eval_id == eval_id) {
      if (out_iteration_index) *out_iteration_index = m.iteration_index;
      if (out_optimized_x) *out_optimized_x = m.optimized_x;
      if (out_video_path) *out_video_path = m.video_path;
      if (out_cost) *out_cost = m.cost;
      return true;
    }
  }
  return false;
}

bool GetIterationEvalInfos(
    int iteration_index,
    int max_count,
    std::vector<EvalInfo>* out_infos) {
  if (!out_infos) return false;
  std::lock_guard<std::mutex> lk(g_meta_mu);
  out_infos->clear();
  for (const auto& m : g_eval_meta) {
    if (m.iteration_index != iteration_index) continue;
    EvalInfo info;
    info.eval_id = m.eval_id;
    info.cost = m.cost;
    info.optimized_x = m.optimized_x;
    out_infos->push_back(std::move(info));
    if (max_count > 0 && (int)out_infos->size() >= max_count) break;
  }
  return !out_infos->empty();
}

class ScopedSensorCallback {
 public:
  explicit ScopedSensorCallback(Task* t) { g_task_for_callback = t; mjcb_sensor = &ScopedSensorCallback::Cb; }
  ~ScopedSensorCallback() { mjcb_sensor = nullptr; g_task_for_callback = nullptr; }
  static void Cb(const mjModel* m, mjData* d, int stage) {
    if (stage == mjSTAGE_ACC) g_task_for_callback->Residual(m, d, d->sensordata);
  }
};

// Return true if the head body is in contact with the ground (worldbody).
static bool HeadTouchedGround(const mjModel* model, const mjData* data) {
  int head_site = mj_name2id(model, mjOBJ_SITE, "head");
  if (head_site < 0) return false;
  int head_body = model->site_bodyid[head_site];
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact* con = data->contact + i;
    int b1 = model->geom_bodyid[con->geom1];
    int b2 = model->geom_bodyid[con->geom2];
    if ((b1 == head_body && b2 == 0) || (b2 == head_body && b1 == 0)) {
      return true;
    }
  }
  return false;
}

Runner::Runner(const std::string& task_name,
               int planner_thread_count,
               double time_limit_seconds)
    : task_(task_name),
      planner_threads_(planner_thread_count),
      time_limit_(time_limit_seconds) {
  agent_ = std::make_unique<Agent>();
  agent_->SetTaskList(GetTasks());
  agent_->gui_task_id = agent_->GetTaskIdByName(task_);
  Agent::LoadModelResult lm = agent_->LoadModel();
  model_ = lm.model.release();
  data_ = mj_makeData(model_);
  int home_id = mj_name2id(model_, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(model_, data_, home_id); else mj_resetData(model_, data_);
  mj_forward(model_, data_);
  agent_->estimator_enabled = false;
  agent_->Initialize(model_);
  agent_->Allocate();
  agent_->Reset(data_->ctrl);
  agent_->plan_enabled = true;
  agent_->action_enabled = true;
  agent_->SetModeByName("Scramble");
  agent_->ActiveTask()->UpdateResidual();
  agent_->SetWeightByName("Orientation", 0.0);
  agent_->SetWeightByName("Angmom", 0.0);
  agent_->SetWeightByName("FootCost", 0.065);
  agent_->SetWeightByName("Height", 0.0);
  agent_->SetWeightByName("Effort", 0.08);
  agent_->ActiveTask()->UpdateResidual();
  names_ = ListCostTermNames(task_);
  pool_ = std::make_unique<ThreadPool>(std::max(1, planner_threads_));
}

Runner::~Runner() {
  // Ensure no callbacks remain
  mjcb_sensor = nullptr;
  g_task_for_callback = nullptr;
  // Free MuJoCo data/model allocated in constructor
  if (data_) { mj_deleteData(data_); data_ = nullptr; }
  if (model_) { mj_deleteModel(model_); model_ = nullptr; }
}

std::vector<double> Runner::DefaultWeights() const {
  std::vector<double> w;
  if (!model_) return w;
  for (int i = 0; i < model_->nsensor && model_->sensor_type[i] == mjSENS_USER; i++) {
    w.push_back(agent_->ActiveTask()->weight[i]);
  }
  return w;
}

std::vector<std::pair<double,double>> Runner::WeightBounds() const {
  std::vector<std::pair<double,double>> b;
  if (!model_) return b;
  for (int i = 0; i < model_->nsensor && model_->sensor_type[i] == mjSENS_USER; i++) {
    double* s = model_->sensor_user + i * model_->nuser_sensor;
    // override bounds for Position term
    std::string name(model_->names + model_->name_sensoradr[i]);
    if (name == "Position") {
      b.emplace_back(0.20, 0.34);
    } else {
      b.emplace_back(s[2], s[3]);
    }
  }
  return b;
}

double Runner::Evaluate(const std::vector<double>& weights,
                        double success_radius_m,
                        bool verbose_progress,
                        bool pace_realtime) {
  for (size_t i = 0; i < names_.size() && i < weights.size(); ++i) agent_->SetWeightByName(names_[i], weights[i]);
  agent_->ActiveTask()->UpdateResidual();

  int home_id_v = mj_name2id(model_, mjOBJ_KEY, "home");
  if (home_id_v >= 0) mj_resetDataKeyframe(model_, data_, home_id_v); else mj_resetData(model_, data_);
  mj_forward(model_, data_);
  agent_->Reset(data_->ctrl);
  agent_->SetModeByName("Scramble");
  agent_->ActiveTask()->UpdateResidual();
  ScopedSensorCallback cb(agent_->ActiveTask());

  std::atomic<bool> exitrequest(false);
  std::atomic<int> uiloadrequest(0);
  std::thread plan_thread([this, &exitrequest, &uiloadrequest]() {
    agent_->Plan(exitrequest, uiloadrequest);
  });

  int total_steps = std::max(1, (int)std::ceil(time_limit_ / model_->opt.timestep));
  double cost = 0.0;
  auto start_wall = std::chrono::steady_clock::now();
  double start_sim = data_->time;
  int last_print_sec_eval = (int)std::floor(data_->time);
  std::vector<int> seconds_marks_eval;
  bool episode_success = false;
  // record initial head x-position for early-fall distance check
  double init_head_x = 0.0;
  {
    int head_site = mj_name2id(model_, mjOBJ_SITE, "head");
    if (head_site >= 0) {
      init_head_x = data_->site_xpos[3 * head_site + 0];
    } else {
      int head_body = mj_name2id(model_, mjOBJ_BODY, "head");
      if (head_body >= 0) init_head_x = data_->xpos[3 * head_body + 0];
      else {
        int trunk = mj_name2id(model_, mjOBJ_BODY, "trunk");
        if (trunk >= 0) init_head_x = data_->xpos[3 * trunk + 0];
      }
    }
  }
  for (int i = 0; i < total_steps; ++i) {
    agent_->ActiveTask()->Transition(model_, data_);
    agent_->state.Set(model_, data_);
    agent_->ActivePlanner().ActionFromPolicy(
        data_->ctrl, agent_->state.state().data(), agent_->state.time(), false);
    mj_step(model_, data_);
    if (pace_realtime) {
      auto target = start_wall + std::chrono::duration<double>(data_->time - start_sim);
      auto now = std::chrono::steady_clock::now();
      if (now < target) std::this_thread::sleep_until(target);
    }
    // check for fall (head contact with ground)
    if (HeadTouchedGround(model_, data_)) {
      double C_fall = 20000.0;
      double k = 10.0;
      double remaining = mju_max(0.0, time_limit_ - (data_->time - start_sim));
      cost += C_fall + k * remaining;
      g_last_fall.store(true);
      g_last_fall_time = (data_->time - start_sim);
      // compute head x-displacement from initial position at fall
      int head_site = mj_name2id(model_, mjOBJ_SITE, "head");
      if (head_site >= 0) {
        double head_x = data_->site_xpos[3 * head_site + 0];
        g_last_fall_dx = std::fabs(head_x - init_head_x);
      } else {
        int head_body = mj_name2id(model_, mjOBJ_BODY, "head");
        if (head_body >= 0) {
          double head_x = data_->xpos[3 * head_body + 0];
          g_last_fall_dx = std::fabs(head_x - init_head_x);
        } else {
          int trunk = mj_name2id(model_, mjOBJ_BODY, "trunk");
          if (trunk >= 0) {
            double trunk_x = data_->xpos[3 * trunk + 0];
            g_last_fall_dx = std::fabs(trunk_x - init_head_x);
          } else {
            g_last_fall_dx = 0.0;
          }
        }
      }
      RecordEvaluation(true);
      break;
    }

    cost += agent_->ActiveTask()->CostValue(data_->sensordata);
    if ((int)std::floor(data_->time) > last_print_sec_eval) {
      last_print_sec_eval = (int)std::floor(data_->time);
      seconds_marks_eval.push_back(last_print_sec_eval);
    }
    bool success = false;
    for (int si = 0, off = 0; si < model_->nsensor && model_->sensor_type[si] == mjSENS_USER; ++si) {
      std::string name(model_->names + model_->name_sensoradr[si]);
      if (name == "Position" && off + 2 < model_->nsensordata) {
        double dx = data_->sensordata[off + 0];
        double dy = data_->sensordata[off + 1];
        double dz = data_->sensordata[off + 2];
        if (std::sqrt(dx*dx + dy*dy + dz*dz) < success_radius_m) success = true;
        break;
      }
      off += model_->sensor_dim[si];
    }
    if (!success) {
      int trunk = mj_name2id(model_, mjOBJ_BODY, "trunk");
      if (trunk >= 0) {
        const double* p = data_->xpos + 3 * trunk;
        const double* g = data_->mocap_pos;
        double dx = p[0] - g[0];
        double dy = p[1] - g[1];
        double dz = p[2] - g[2];
        if (std::sqrt(dx*dx + dy*dy + dz*dz) < success_radius_m) success = true;
      }
    }
    if (success) { episode_success = true; break; }
  }

  exitrequest.store(true);
  if (plan_thread.joinable()) plan_thread.join();
  // do not print seconds in bulk; keep output minimal
  if (!g_last_fall.load()) RecordEvaluation(false);

  // Add terminal penalty: K_target * distance(head, target) if target not reached
  if (!episode_success) {
    // Determine head position (prefer site 'head', fallback to body 'head', else trunk)
    double head_pos[3] = {0.0, 0.0, 0.0};
    bool have_head = false;
    int head_site = mj_name2id(model_, mjOBJ_SITE, "head");
    if (head_site >= 0) {
      const double* sx = data_->site_xpos + 3 * head_site;
      head_pos[0] = sx[0]; head_pos[1] = sx[1]; head_pos[2] = sx[2];
      have_head = true;
    } else {
      int head_body = mj_name2id(model_, mjOBJ_BODY, "head");
      if (head_body >= 0) {
        const double* bx = data_->xpos + 3 * head_body;
        head_pos[0] = bx[0]; head_pos[1] = bx[1]; head_pos[2] = bx[2];
        have_head = true;
      }
    }
    if (!have_head) {
      int trunk = mj_name2id(model_, mjOBJ_BODY, "trunk");
      if (trunk >= 0) {
        const double* bx = data_->xpos + 3 * trunk;
        head_pos[0] = bx[0]; head_pos[1] = bx[1]; head_pos[2] = bx[2];
        have_head = true;
      }
    }
    const double* target = data_->mocap_pos;  // assumed target position
    double dist = 0.0;
    if (have_head && target) {
      double dx = head_pos[0] - target[0];
      double dy = head_pos[1] - target[1];
      double dz = head_pos[2] - target[2];
      dist = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    const double K_target = 10000.0;
    cost += K_target * dist;
  }

  return cost;
}

namespace {
static bool g_mj_gl_error = false;
static void OnMjError(const char* msg) { g_mj_gl_error = true; }
static void OnMjWarning(const char* msg) {}
}

bool Runner::RenderVideo(const std::vector<double>& weights,
                         const std::string& out_dir,
                         int width,
                         int height,
                         double fps,
                         double duration_seconds,
                         std::string* out_video_path,
                         const std::string& out_basename,
                         bool pace_realtime) {
  // minimal logging for video
  for (size_t i = 0; i < names_.size() && i < weights.size(); ++i) agent_->SetWeightByName(names_[i], weights[i]);
  agent_->ActiveTask()->UpdateResidual();
  // Ensure a fresh episode for every RenderVideo call.
  int home_id_v = mj_name2id(model_, mjOBJ_KEY, "home");
  if (home_id_v >= 0) mj_resetDataKeyframe(model_, data_, home_id_v); else mj_resetData(model_, data_);
  mj_forward(model_, data_);
  agent_->Reset(data_->ctrl);
  agent_->SetModeByName("Scramble");
  agent_->ActiveTask()->UpdateResidual();
  // ensure residual sensors are populated during physics stepping
  ScopedSensorCallback sensor_cb(agent_->ActiveTask());

  int total_frames = std::max(1, (int)std::round(duration_seconds * fps));
  double dt = 1.0 / fps;
  double next_shot = 0.0;

  // Defer starting the planner thread until after GL context is created,
  // so any early returns do not leave a joinable thread.
  std::atomic<bool> exitrequest(false);
  std::atomic<int> uiloadrequest(0);
  std::thread plan_thread;

  std::vector<unsigned char> rgb(3 * width * height);
  mjrContext con; mjr_defaultContext(&con);
  mjvScene scn; mjv_defaultScene(&scn);
  mjvOption opt; mjv_defaultOption(&opt);
  for (int i = 0; i < mjNGROUP; ++i) {
    opt.geomgroup[i] = 1;
    opt.sitegroup[i] = 1;
  }
  mjvPerturb pert; mjv_defaultPerturb(&pert);
  mjvCamera cam; mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE;
  cam.fixedcamid = -1;
  cam.trackbodyid = -1;
  cam.azimuth = model_->vis.global.azimuth;
  cam.elevation = model_->vis.global.elevation;
  cam.distance = model_->stat.extent;
  cam.lookat[0] = model_->stat.center[0];
  cam.lookat[1] = model_->stat.center[1];
  cam.lookat[2] = model_->stat.center[2];

  GLFWwindow* offscreen = nullptr;
  std::string backend;
  if (const char* gl = std::getenv("MUJOCO_GL")) {
    backend.assign(gl);
    for (auto& c : backend) c = std::tolower(c);
  }

  bool use_glfw_hidden = (backend.empty() || backend == "glfw");
  if (use_glfw_hidden) {
    if (!glfwInit()) { std::cout << "RenderVideo: glfwInit failed" << std::endl; return false; }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_NO_ERROR, GLFW_TRUE);
    offscreen = glfwCreateWindow(std::max(1, width), std::max(1, height), "offscreen", nullptr, nullptr);
    if (!offscreen) { std::cout << "RenderVideo: glfwCreateWindow failed" << std::endl; glfwTerminate(); return false; }
    glfwMakeContextCurrent(offscreen);
  }

  void (*prev_err)(const char*) = mju_user_error;
  void (*prev_warn)(const char*) = mju_user_warning;
  g_mj_gl_error = false;
  mju_user_error = OnMjError;
  mju_user_warning = OnMjWarning;
  mjr_makeContext(model_, &con, mjFONTSCALE_150);
  if (g_mj_gl_error) {
    std::cout << "RenderVideo: context error" << std::endl;
    mju_user_error = prev_err;
    mju_user_warning = prev_warn;
    if (offscreen) { glfwDestroyWindow(offscreen); glfwTerminate(); }
    return false;
  }
  mjv_makeScene(model_, &scn, 2000);

  // Prepare ffmpeg pipe to avoid saving intermediate frames to disk.
  std::string out = out_dir + "/" + out_basename + ".mp4";
  std::string ff_cmd =
      std::string("ffmpeg -hide_banner -loglevel error -nostats -y ") +
      "-f rawvideo -pix_fmt rgb24 " +
      "-s " + std::to_string(std::max(1, width)) + "x" + std::to_string(std::max(1, height)) + " " +
      "-r " + std::to_string((int)std::round(fps)) + " -i - " +
      "-c:v libx264 -pix_fmt yuv420p " + out + " 1>/dev/null 2>&1";
  FILE* ff = popen(ff_cmd.c_str(), "w");
  if (!ff) {
    mju_user_error = prev_err;
    mju_user_warning = prev_warn;
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    if (offscreen) { glfwDestroyWindow(offscreen); glfwTerminate(); }
    return false;
  }

  // suppressed: loop frames logging
  int step_count = 0;
  auto start_wall = std::chrono::steady_clock::now();
  double start_sim = data_->time;
  int last_print_sec = (int)std::floor(data_->time);
  std::vector<int> seconds_marks_video;
  bool fell = false;

  // Start background planner now that GL context is valid.
  plan_thread = std::thread([this, &exitrequest, &uiloadrequest]() {
    agent_->Plan(exitrequest, uiloadrequest);
  });
  // do not delay; match evaluation rollout timing as closely as possible
  next_shot = data_->time;  // should be initial time (e.g. 0)
  for (int f = 0; f < total_frames; ++f) {
    while (data_->time < next_shot) {
      agent_->ActiveTask()->Transition(model_, data_);
      agent_->state.Set(model_, data_);
      agent_->ActivePlanner().ActionFromPolicy(
          data_->ctrl, agent_->state.state().data(), agent_->state.time(), false);
      mj_step(model_, data_);
      // terminate video rollout early on head-ground contact
      if (HeadTouchedGround(model_, data_)) { fell = true; break; }
      if (pace_realtime) {
        auto target = start_wall + std::chrono::duration<double>(data_->time - start_sim);
        auto now = std::chrono::steady_clock::now();
        if (now < target) std::this_thread::sleep_until(target);
      }
      step_count++;
    }
    if (fell) break;  // stop immediately; do not render/save further frames
    if ((int)std::floor(data_->time) > last_print_sec) {
      last_print_sec = (int)std::floor(data_->time);
      seconds_marks_video.push_back(last_print_sec);
    }
    next_shot += dt;

    // suppressed frame-by-frame logs
    mjv_updateScene(model_, data_, &opt, &pert, &cam, mjCAT_ALL, &scn);
    mjrRect rect; rect.left = 0; rect.bottom = 0; rect.width = width; rect.height = height;
    mjr_render(rect, &scn, &con);
    mjr_readPixels(rgb.data(), nullptr, rect, &con);

    // flip vertically so the video matches MJPC GUI orientation
    for (int y = 0; y < height / 2; ++y) {
      unsigned char* row_top = rgb.data() + 3 * width * y;
      unsigned char* row_bot = rgb.data() + 3 * width * (height - 1 - y);
      for (int x = 0; x < 3 * width; ++x) std::swap(row_top[x], row_bot[x]);
    }

    size_t expected = (size_t)3 * (size_t)width * (size_t)height;
    size_t wrote = fwrite(rgb.data(), 1, expected, ff);
    if (wrote != expected) {
      std::cout << "RenderVideo: failed to write frame to ffmpeg pipe" << std::endl;
      fell = true;  // force exit
      break;
    }
  }

  mjv_freeScene(&scn);
  mjr_freeContext(&con);

  mju_user_error = prev_err;
  mju_user_warning = prev_warn;

  // stop background planner
  exitrequest.store(true);
  if (plan_thread.joinable()) plan_thread.join();
  // do not print seconds in bulk for video either

  if (offscreen) { glfwDestroyWindow(offscreen); glfwTerminate(); }

  int status = pclose(ff);
  bool ok = false;
  if (status != -1) {
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) ok = true;
  }
  if (out_video_path) *out_video_path = out;
  return ok;
}

double EvaluateCostForWeights(
    const std::string& task_name,
    const std::vector<std::pair<std::string, double>>& name_value_weights,
    int planner_thread_count,
    double total_time_seconds,
    bool verbose_progress) {
  Agent agent;
  agent.SetTaskList(GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::cerr << "Unknown task: " << task_name << "\n";
    return -1.0;
  }

  Agent::LoadModelResult load_model = agent.LoadModel();
  mjModel* model = load_model.model.get();
  if (!model) {
    std::cerr << load_model.error << "\n";
    return -1.0;
  }
  mjData* data = mj_makeData(model);

  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) {
    mj_resetDataKeyframe(model, data, home_id);
  } else {
    mj_resetData(model, data);
  }
  mj_forward(model, data);

  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);

  for (const auto& nv : name_value_weights) {
    if (agent.SetWeightByName(nv.first, nv.second) == -1) {
      std::cerr << "Warning: weight name not found: " << nv.first << "\n";
    }
  }
  agent.ActiveTask()->UpdateResidual();

  ScopedSensorCallback cb(agent.ActiveTask());

  ThreadPool pool(std::max(1, planner_thread_count));

  int total_steps = std::max(1, (int)std::ceil(total_time_seconds / model->opt.timestep));
  int current_sec = (int)std::floor(data->time);
  double accumulated_cost = 0.0;

  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);

    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(), agent.state.time(),
        /*use_previous=*/false);

    mj_step(model, data);
    accumulated_cost += agent.ActiveTask()->CostValue(data->sensordata);

    agent.PlanIteration(&pool);

    if (verbose_progress && std::floor(data->time) > current_sec) {
      current_sec = (int)std::floor(data->time);
      std::cout << "t=" << current_sec
                << "s, step_cost=" << agent.ActiveTask()->CostValue(data->sensordata)
                << "\n";
    }
  }

  mj_deleteData(data);
  return accumulated_cost;
}

std::vector<std::string> ListCostTermNames(const std::string& task_name) {
  Agent agent;
  agent.SetTaskList(GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) return {};
  Agent::LoadModelResult load_model = agent.LoadModel();
  mjModel* model = load_model.model.get();
  if (!model) return {};

  std::vector<std::string> names;
  for (int i = 0; i < model->nsensor && model->sensor_type[i] == mjSENS_USER; i++) {
    std::string name(model->names + model->name_sensoradr[i]);
    names.push_back(name);
  }
  return names;
}

std::vector<std::pair<std::string,double>> DefaultCostWeights(const std::string& task_name) {
  Agent agent;
  agent.SetTaskList(GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  Agent::LoadModelResult lm = agent.LoadModel();
  mjModel* m = lm.model.get();
  std::vector<std::pair<std::string,double>> out;
  if (!m) return out;
  for (int i = 0; i < m->nsensor && m->sensor_type[i] == mjSENS_USER; ++i) {
    std::string n(m->names + m->name_sensoradr[i]);
    out.emplace_back(n, agent.ActiveTask()->weight[i]);
  }
  return out;
}

}  // namespace weights_opt
}  // namespace mjpc


