#include "dune_ppo_training_utils_internal.h"
#include "open_spiel/utils/json.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <unordered_set>

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <c10/core/Event.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace open_spiel {

namespace ppo_utils_internal {
// WO-PERF-1 (R2 review fix F2): the "first TrainPpoUpdate in this process"
// marker is keyed PER MODEL (module identity), not process-global, so a
// process that trains two models (e.g. a pilot arm and a live arm) force-
// measures each model's own first update. A resume is always a new process,
// so first-call-per-model-in-process is exactly the "startup/resume"
// condition the cadenced pre-pass must always measure on. Only touched in
// --diag_prepass_mode=cadenced; full mode never reaches it.
std::mutex g_diag_prepass_seen_mu;
std::unordered_set<const void*> g_diag_prepass_seen_models;
// Returns true exactly once per model identity per process (per reset).
bool ConsumeFirstPrepassForModel(const void* model_identity) {
  std::lock_guard<std::mutex> lock(g_diag_prepass_seen_mu);
  return g_diag_prepass_seen_models.insert(model_identity).second;
}
}  // namespace ppo_utils_internal

void ResetDiagPrepassStateForTesting() {
  std::lock_guard<std::mutex> lock(ppo_utils_internal::g_diag_prepass_seen_mu);
  ppo_utils_internal::g_diag_prepass_seen_models.clear();
}

void V5LegacySlotResidualState::Initialize(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model, int anneal_updates) {
  active = false;
  N = anneal_updates;
  if (!model || !model->policy_head) return;
  torch::NoGradGuard no_grad;
  init_w_residuals.clear();
  init_b_residuals.clear();
  float max_abs_res = 0.0f;
  for (const auto& fam : families) {
    std::vector<torch::Tensor> w_rows;
    for (int64_t s : fam) {
      w_rows.push_back(model->policy_head->weight[s]);
    }
    torch::Tensor w_stack = torch::stack(w_rows);
    torch::Tensor bar_w = w_stack.mean(0);
    torch::Tensor res_w = w_stack - bar_w.unsqueeze(0);
    max_abs_res = std::max(max_abs_res, res_w.abs().max().item<float>());
    init_w_residuals.push_back(res_w.clone());

    if (model->policy_head->bias.defined()) {
      std::vector<torch::Tensor> b_rows;
      for (int64_t s : fam) {
        b_rows.push_back(model->policy_head->bias[s]);
      }
      torch::Tensor b_stack = torch::stack(b_rows);
      torch::Tensor bar_b = b_stack.mean(0);
      torch::Tensor res_b = b_stack - bar_b.unsqueeze(0);
      max_abs_res = std::max(max_abs_res, res_b.abs().max().item<float>());
      init_b_residuals.push_back(res_b.clone());
    }
  }
  if (anneal_updates > 0 || max_abs_res <= 1e-5f) {
    active = true;
    if (anneal_updates <= 0 && max_abs_res > 0.0f) {
      Apply(model, 1);
    }
  }
}

void V5LegacySlotResidualState::Apply(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model, int completed_updates_in_run) {
  if (!active || !model || !model->policy_head) return;
  torch::NoGradGuard no_grad;
  const double alpha = (N <= 0 || completed_updates_in_run >= N)
      ? 0.0
      : std::max(0.0, 1.0 - static_cast<double>(completed_updates_in_run) / static_cast<double>(N));
  for (size_t f = 0; f < families.size(); ++f) {
    const auto& fam = families[f];
    std::vector<torch::Tensor> w_rows;
    for (int64_t s : fam) {
      w_rows.push_back(model->policy_head->weight[s]);
    }
    torch::Tensor cur_bar_w = torch::stack(w_rows).mean(0);
    const auto& res_w = init_w_residuals[f];
    for (size_t i = 0; i < fam.size(); ++i) {
      int64_t s = fam[i];
      if (alpha == 0.0) {
        model->policy_head->weight[s].copy_(cur_bar_w);
      } else {
        model->policy_head->weight[s].copy_(cur_bar_w + static_cast<float>(alpha) * res_w[i]);
      }
    }
    if (model->policy_head->bias.defined() && f < init_b_residuals.size()) {
      std::vector<torch::Tensor> b_rows;
      for (int64_t s : fam) {
        b_rows.push_back(model->policy_head->bias[s]);
      }
      torch::Tensor cur_bar_b = torch::stack(b_rows).mean(0);
      const auto& res_b = init_b_residuals[f];
      for (size_t i = 0; i < fam.size(); ++i) {
        int64_t s = fam[i];
        if (alpha == 0.0) {
          model->policy_head->bias[s].copy_(cur_bar_b);
        } else {
          model->policy_head->bias[s].copy_(cur_bar_b + static_cast<float>(alpha) * res_b[i]);
        }
      }
    }
  }
  if (model->bf16_inference_cache_ready()) {
    model->RefreshBf16InferenceCache();
  }
}

bool V5LegacySlotResidualState::SaveSidecar(const std::string& path) const {
  if (!active || N <= 0 || init_w_residuals.size() != families.size()) {
    return false;
  }
  const std::string tmp_path = path + ".tmp";
  std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
  if (!out) return false;

  const int64_t hidden_dim =
      init_w_residuals.empty() ? 0 : init_w_residuals[0].size(1);
  out << "{\"version\":1"
      << ",\"start_global_update\":" << start_global_update
      << ",\"N\":" << N
      << ",\"hidden_dim\":" << hidden_dim
      << ",\"has_bias\":" << (!init_b_residuals.empty() ? "true" : "false")
      << ",\"init_w_residuals\":[";
  for (size_t f = 0; f < init_w_residuals.size(); ++f) {
    if (f > 0) out << ",";
    torch::Tensor w_cpu =
        init_w_residuals[f].detach().to(torch::kCPU, torch::kFloat32).contiguous();
    const float* wp = w_cpu.data_ptr<float>();
    const int64_t numel = w_cpu.numel();
    out << "[";
    for (int64_t i = 0; i < numel; ++i) {
      if (i > 0) out << ",";
      out << FormatDoubleRoundTrip(static_cast<double>(wp[i]));
    }
    out << "]";
  }
  out << "],\"init_b_residuals\":[";
  for (size_t f = 0; f < init_b_residuals.size(); ++f) {
    if (f > 0) out << ",";
    torch::Tensor b_cpu =
        init_b_residuals[f].detach().to(torch::kCPU, torch::kFloat32).contiguous();
    const float* bp = b_cpu.data_ptr<float>();
    const int64_t numel = b_cpu.numel();
    out << "[";
    for (int64_t i = 0; i < numel; ++i) {
      if (i > 0) out << ",";
      out << FormatDoubleRoundTrip(static_cast<double>(bp[i]));
    }
    out << "]";
  }
  out << "]}\n";
  out.flush();
  if (!out.good()) {
    out.close();
    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  out.close();
  std::error_code ec;
  std::filesystem::rename(tmp_path, path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  return true;
}

bool V5LegacySlotResidualState::LoadSidecar(const std::string& path,
                                            torch::Device device) {
  if (!std::filesystem::exists(path)) return false;
  std::ifstream ifs(path);
  if (!ifs) return false;
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = json::FromString(content);
  if (!val_opt.has_value() || !val_opt->IsObject()) return false;
  const auto& obj = val_opt->GetObject();

  auto it_ver = obj.find("version");
  auto it_start = obj.find("start_global_update");
  auto it_n = obj.find("N");
  auto it_hd = obj.find("hidden_dim");
  auto it_w = obj.find("init_w_residuals");
  auto it_b = obj.find("init_b_residuals");
  if (it_ver == obj.end() || !it_ver->second.IsInt() ||
      it_ver->second.GetInt() != 1 ||
      it_start == obj.end() || !it_start->second.IsInt() ||
      it_n == obj.end() || !it_n->second.IsInt() ||
      it_hd == obj.end() || !it_hd->second.IsInt() ||
      it_w == obj.end() || !it_w->second.IsArray()) {
    return false;
  }
  const int64_t hidden_dim = it_hd->second.GetInt();
  const auto& w_arr = it_w->second.GetArray();
  if (w_arr.size() != families.size() || hidden_dim <= 0) return false;

  std::vector<torch::Tensor> loaded_w;
  loaded_w.reserve(families.size());
  for (size_t f = 0; f < families.size(); ++f) {
    if (!w_arr[f].IsArray()) return false;
    const auto& fam_arr = w_arr[f].GetArray();
    const int64_t rows = static_cast<int64_t>(families[f].size());
    if (static_cast<int64_t>(fam_arr.size()) != rows * hidden_dim) return false;
    torch::Tensor t = torch::empty(
        {rows, hidden_dim}, torch::TensorOptions().dtype(torch::kFloat32));
    float* tp = t.data_ptr<float>();
    for (size_t i = 0; i < fam_arr.size(); ++i) {
      if (fam_arr[i].IsDouble()) {
        tp[i] = static_cast<float>(fam_arr[i].GetDouble());
      } else if (fam_arr[i].IsInt()) {
        tp[i] = static_cast<float>(fam_arr[i].GetInt());
      } else {
        return false;
      }
    }
    loaded_w.push_back(t.to(device));
  }

  std::vector<torch::Tensor> loaded_b;
  if (it_b != obj.end() && it_b->second.IsArray() &&
      !it_b->second.GetArray().empty()) {
    const auto& b_arr = it_b->second.GetArray();
    if (b_arr.size() != families.size()) return false;
    loaded_b.reserve(families.size());
    for (size_t f = 0; f < families.size(); ++f) {
      if (!b_arr[f].IsArray()) return false;
      const auto& fam_arr = b_arr[f].GetArray();
      const int64_t rows = static_cast<int64_t>(families[f].size());
      if (static_cast<int64_t>(fam_arr.size()) != rows) return false;
      torch::Tensor t =
          torch::empty({rows}, torch::TensorOptions().dtype(torch::kFloat32));
      float* tp = t.data_ptr<float>();
      for (size_t i = 0; i < fam_arr.size(); ++i) {
        if (fam_arr[i].IsDouble()) {
          tp[i] = static_cast<float>(fam_arr[i].GetDouble());
        } else if (fam_arr[i].IsInt()) {
          tp[i] = static_cast<float>(fam_arr[i].GetInt());
        } else {
          return false;
        }
      }
      loaded_b.push_back(t.to(device));
    }
  }

  start_global_update = static_cast<int>(it_start->second.GetInt());
  N = static_cast<int>(it_n->second.GetInt());
  init_w_residuals = std::move(loaded_w);
  init_b_residuals = std::move(loaded_b);
  active = true;
  return true;
}

namespace {
template <typename T>
bool WritePod(std::ostream& out, const T& val) {
  out.write(reinterpret_cast<const char*>(&val), sizeof(T));
  return out.good();
}

template <typename T>
bool ReadPod(std::istream& in, T* val) {
  in.read(reinterpret_cast<char*>(val), sizeof(T));
  return static_cast<bool>(in);
}

bool WriteStringBin(std::ostream& out, const std::string& s) {
  uint64_t len = static_cast<uint64_t>(s.size());
  if (!WritePod(out, len)) return false;
  if (len > 0) {
    out.write(s.data(), static_cast<std::streamsize>(len));
  }
  return out.good();
}

bool ReadStringBin(std::istream& in, std::string* s) {
  uint64_t len = 0;
  if (!ReadPod(in, &len)) return false;
  if (len > (1ULL << 28)) return false;
  s->resize(static_cast<size_t>(len));
  if (len > 0) {
    in.read(s->data(), static_cast<std::streamsize>(len));
  }
  return static_cast<bool>(in);
}

template <typename T>
bool WriteVecPod(std::ostream& out, const std::vector<T>& v) {
  uint64_t len = static_cast<uint64_t>(v.size());
  if (!WritePod(out, len)) return false;
  if (len > 0) {
    out.write(reinterpret_cast<const char*>(v.data()),
              static_cast<std::streamsize>(len * sizeof(T)));
  }
  return out.good();
}

template <typename T>
bool ReadVecPod(std::istream& in, std::vector<T>* v) {
  uint64_t len = 0;
  if (!ReadPod(in, &len)) return false;
  if (len > (1ULL << 28)) return false;
  v->resize(static_cast<size_t>(len));
  if (len > 0) {
    in.read(reinterpret_cast<char*>(v->data()),
            static_cast<std::streamsize>(len * sizeof(T)));
  }
  return static_cast<bool>(in);
}

bool WriteCandidateDataBin(std::ostream& out,
                           const dune_semantic::CandidateActionData& cad) {
  if (!WriteVecPod(out, cad.actions)) return false;
  if (!WriteVecPod(out, cad.features)) return false;
  if (!WriteVecPod(out, cad.card_ids)) return false;
  if (!WriteVecPod(out, cad.space_ids)) return false;
  if (!WriteVecPod(out, cad.supported)) return false;
  std::vector<uint8_t> role_bytes(cad.roles.size());
  for (size_t i = 0; i < cad.roles.size(); ++i) {
    role_bytes[i] = static_cast<uint8_t>(cad.roles[i]);
  }
  if (!WriteVecPod(out, role_bytes)) return false;
  if (!WriteVecPod(out, cad.v5_item_ids)) return false;
  if (!WriteVecPod(out, cad.v5_families)) return false;
  if (!WriteVecPod(out, cad.v5_ctx_features)) return false;
  return true;
}

bool ReadCandidateDataBin(std::istream& in,
                          dune_semantic::CandidateActionData* cad) {
  cad->clear();
  if (!ReadVecPod(in, &cad->actions)) return false;
  if (!ReadVecPod(in, &cad->features)) return false;
  if (!ReadVecPod(in, &cad->card_ids)) return false;
  if (!ReadVecPod(in, &cad->space_ids)) return false;
  if (!ReadVecPod(in, &cad->supported)) return false;
  std::vector<uint8_t> role_bytes;
  if (!ReadVecPod(in, &role_bytes)) return false;
  cad->roles.resize(role_bytes.size());
  for (size_t i = 0; i < role_bytes.size(); ++i) {
    cad->roles[i] = static_cast<dune_semantic::ActionRole>(role_bytes[i]);
  }
  if (!ReadVecPod(in, &cad->v5_item_ids)) return false;
  if (!ReadVecPod(in, &cad->v5_families)) return false;
  if (!ReadVecPod(in, &cad->v5_ctx_features)) return false;
  return true;
}
}  // namespace

bool R765PreserveWindowGuard::SaveState(std::ostream& out) const {
  const char kMagic[8] = {'R', '7', '6', '5', 'P', 'W', 'G', '1'};
  out.write(kMagic, 8);
  uint64_t n = static_cast<uint64_t>(window_.size());
  if (!WritePod(out, n)) return false;
  for (const auto& entry : window_) {
    if (!WritePod(out, entry.preserved)) return false;
    if (!WritePod(out, entry.eligible)) return false;
  }
  return out.good();
}

bool R765PreserveWindowGuard::LoadState(std::istream& in) {
  char magic[8] = {};
  in.read(magic, 8);
  const char kExpected[8] = {'R', '7', '6', '5', 'P', 'W', 'G', '1'};
  if (!in || std::memcmp(magic, kExpected, 8) != 0) return false;
  uint64_t n = 0;
  if (!ReadPod(in, &n) || n > 1024) return false;
  std::deque<WindowEntry> loaded;
  for (uint64_t i = 0; i < n; ++i) {
    WindowEntry e;
    if (!ReadPod(in, &e.preserved) || !ReadPod(in, &e.eligible)) return false;
    loaded.push_back(e);
  }
  window_ = std::move(loaded);
  return true;
}

bool R765PreserveWindowGuard::SaveState(const std::string& path) const {
  const std::string tmp_path = path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
  if (!out || !SaveState(out)) {
    out.close();
    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  out.flush();
  out.close();
  std::error_code ec;
  std::filesystem::rename(tmp_path, path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  return true;
}

bool R765PreserveWindowGuard::LoadState(const std::string& path) {
  if (!std::filesystem::exists(path)) return false;
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  return LoadState(in);
}

std::vector<uint8_t> R765ReplayCohortManager::FreshCohortVisitedMask(
    int global_update, size_t fresh_count,
    const std::vector<uint8_t>& batch_visited_mask) const {
  std::vector<uint8_t> out(fresh_count, 0);
  const size_t n = std::min(last_batch_map_.size(), batch_visited_mask.size());
  for (size_t i = 0; i < n; ++i) {
    if (batch_visited_mask[i] != 0 &&
        last_batch_map_[i].birth_update == global_update &&
        last_batch_map_[i].row_idx_in_cohort < fresh_count) {
      out[last_batch_map_[i].row_idx_in_cohort] = 1;
    }
  }
  return out;
}

bool R765ReplayCohortManager::SaveState(
    const std::string& path,
    const R765PreserveWindowGuard* preserve_guard) const {
  const std::string tmp_path = path + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
  if (!out) return false;

  const char kMagic[8] = {'R', '7', '6', '5', 'R', 'P', 'L', 'Y'};
  out.write(kMagic, 8);
  const uint32_t version = 1;
  if (!WritePod(out, version)) return false;

  const uint64_t num_tc = static_cast<uint64_t>(train_cohorts_.size());
  if (!WritePod(out, num_tc)) return false;
  for (const auto& tc : train_cohorts_) {
    int32_t birth = tc.birth_update;
    int32_t age = tc.age;
    uint64_t num_ex = static_cast<uint64_t>(tc.examples.size());
    if (!WritePod(out, birth) || !WritePod(out, age) || !WritePod(out, num_ex)) {
      return false;
    }
    for (const auto& ex : tc.examples) {
      int32_t role_i = static_cast<int32_t>(ex.role);
      uint8_t vt_att = ex.value_target_attached ? 1 : 0;
      if (!WriteVecPod(out, ex.observation) ||
          !WritePod(out, ex.player) ||
          !WriteVecPod(out, ex.legal_actions) ||
          !WriteVecPod(out, ex.normalized_visits) ||
          !WritePod(out, ex.value_target) ||
          !WritePod(out, vt_att) ||
          !WriteStringBin(out, ex.checkpoint_hash) ||
          !WritePod(out, ex.update_id) ||
          !WritePod(out, ex.episode_id) ||
          !WritePod(out, ex.decision_id) ||
          !WritePod(out, ex.simulations_completed) ||
          !WritePod(out, ex.num_covered_actions) ||
          !WritePod(out, ex.covered_prior_mass) ||
          !WritePod(out, ex.mean_depth) ||
          !WritePod(out, ex.terminal_leaf_fraction) ||
          !WritePod(out, role_i) ||
          !WriteCandidateDataBin(out, ex.candidate_data) ||
          !WritePod(out, ex.raw_action) ||
          !WritePod(out, ex.best_action) ||
          !WritePod(out, ex.replay_age) ||
          !WritePod(out, ex.row_loss_weight) ||
          !WritePod(out, ex.r765_root_uid)) {
        return false;
      }
    }
  }

  const uint64_t num_dc = static_cast<uint64_t>(diag_cohorts_.size());
  if (!WritePod(out, num_dc)) return false;
  for (const auto& dc : diag_cohorts_) {
    int32_t birth = dc.birth_update;
    int32_t exp_age = dc.expected_final_age;
    int32_t last_pres = dc.last_presented_update;
    uint8_t full_cov = dc.final_age_full_coverage ? 1 : 0;
    uint64_t num_probes = static_cast<uint64_t>(dc.probes.size());
    if (!WritePod(out, birth) || !WritePod(out, exp_age) ||
        !WritePod(out, last_pres) || !WritePod(out, full_cov) ||
        !WritePod(out, num_probes)) {
      return false;
    }
    for (const auto& pr : dc.probes) {
      if (!WritePod(out, pr.root_uid) ||
          !WriteVecPod(out, pr.observation) ||
          !WriteVecPod(out, pr.legal_actions) ||
          !WriteCandidateDataBin(out, pr.candidate_data) ||
          !WritePod(out, pr.a_star) ||
          !WritePod(out, pr.a_ref) ||
          !WritePod(out, pr.t0_log_odds)) {
        return false;
      }
    }
    uint64_t num_live = static_cast<uint64_t>(dc.root_d_final_live.size());
    if (!WritePod(out, num_live)) return false;
    for (const auto& kv : dc.root_d_final_live) {
      if (!WritePod(out, kv.first) || !WritePod(out, kv.second)) return false;
    }
  }

  const uint8_t has_guard = (preserve_guard != nullptr) ? 1 : 0;
  if (!WritePod(out, has_guard)) return false;
  if (preserve_guard != nullptr && !preserve_guard->SaveState(out)) {
    return false;
  }

  out.flush();
  if (!out.good()) {
    out.close();
    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  out.close();
  std::error_code ec;
  std::filesystem::rename(tmp_path, path, ec);
  if (ec) {
    std::filesystem::remove(tmp_path, ec);
    return false;
  }
  return true;
}

bool R765ReplayCohortManager::LoadState(
    const std::string& path,
    R765PreserveWindowGuard* preserve_guard) {
  if (!std::filesystem::exists(path)) return false;
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  char magic[8] = {};
  in.read(magic, 8);
  const char kExpected[8] = {'R', '7', '6', '5', 'R', 'P', 'L', 'Y'};
  if (!in || std::memcmp(magic, kExpected, 8) != 0) return false;
  uint32_t version = 0;
  if (!ReadPod(in, &version) || version != 1) return false;

  uint64_t num_tc = 0;
  if (!ReadPod(in, &num_tc) || num_tc > 1024) return false;
  std::vector<TrainCohort> loaded_tc;
  loaded_tc.reserve(static_cast<size_t>(num_tc));
  for (uint64_t c = 0; c < num_tc; ++c) {
    TrainCohort tc;
    int32_t birth = 0, age = 0;
    uint64_t num_ex = 0;
    if (!ReadPod(in, &birth) || !ReadPod(in, &age) || !ReadPod(in, &num_ex) ||
        num_ex > (1ULL << 22)) {
      return false;
    }
    tc.birth_update = birth;
    tc.age = age;
    tc.examples.resize(static_cast<size_t>(num_ex));
    for (uint64_t i = 0; i < num_ex; ++i) {
      auto& ex = tc.examples[static_cast<size_t>(i)];
      int32_t role_i = 0;
      uint8_t vt_att = 0;
      if (!ReadVecPod(in, &ex.observation) ||
          !ReadPod(in, &ex.player) ||
          !ReadVecPod(in, &ex.legal_actions) ||
          !ReadVecPod(in, &ex.normalized_visits) ||
          !ReadPod(in, &ex.value_target) ||
          !ReadPod(in, &vt_att) ||
          !ReadStringBin(in, &ex.checkpoint_hash) ||
          !ReadPod(in, &ex.update_id) ||
          !ReadPod(in, &ex.episode_id) ||
          !ReadPod(in, &ex.decision_id) ||
          !ReadPod(in, &ex.simulations_completed) ||
          !ReadPod(in, &ex.num_covered_actions) ||
          !ReadPod(in, &ex.covered_prior_mass) ||
          !ReadPod(in, &ex.mean_depth) ||
          !ReadPod(in, &ex.terminal_leaf_fraction) ||
          !ReadPod(in, &role_i) ||
          !ReadCandidateDataBin(in, &ex.candidate_data) ||
          !ReadPod(in, &ex.raw_action) ||
          !ReadPod(in, &ex.best_action) ||
          !ReadPod(in, &ex.replay_age) ||
          !ReadPod(in, &ex.row_loss_weight) ||
          !ReadPod(in, &ex.r765_root_uid)) {
        return false;
      }
      ex.value_target_attached = (vt_att != 0);
      ex.role = static_cast<DuneDecisionRole>(role_i);
    }
    loaded_tc.push_back(std::move(tc));
  }

  uint64_t num_dc = 0;
  if (!ReadPod(in, &num_dc) || num_dc > 4096) return false;
  std::vector<DiagCohort> loaded_dc;
  loaded_dc.reserve(static_cast<size_t>(num_dc));
  for (uint64_t c = 0; c < num_dc; ++c) {
    DiagCohort dc;
    int32_t birth = 0, exp_age = 0, last_pres = -1;
    uint8_t full_cov = 0;
    uint64_t num_probes = 0;
    if (!ReadPod(in, &birth) || !ReadPod(in, &exp_age) ||
        !ReadPod(in, &last_pres) || !ReadPod(in, &full_cov) ||
        !ReadPod(in, &num_probes) || num_probes > (1ULL << 22)) {
      return false;
    }
    dc.birth_update = birth;
    dc.expected_final_age = exp_age;
    dc.last_presented_update = last_pres;
    dc.final_age_full_coverage = (full_cov != 0);
    dc.probes.resize(static_cast<size_t>(num_probes));
    for (uint64_t i = 0; i < num_probes; ++i) {
      auto& pr = dc.probes[static_cast<size_t>(i)];
      if (!ReadPod(in, &pr.root_uid) ||
          !ReadVecPod(in, &pr.observation) ||
          !ReadVecPod(in, &pr.legal_actions) ||
          !ReadCandidateDataBin(in, &pr.candidate_data) ||
          !ReadPod(in, &pr.a_star) ||
          !ReadPod(in, &pr.a_ref) ||
          !ReadPod(in, &pr.t0_log_odds)) {
        return false;
      }
    }
    uint64_t num_live = 0;
    if (!ReadPod(in, &num_live) || num_live > (1ULL << 22)) return false;
    for (uint64_t i = 0; i < num_live; ++i) {
      uint64_t uid = 0;
      double val = 0.0;
      if (!ReadPod(in, &uid) || !ReadPod(in, &val)) return false;
      dc.root_d_final_live[uid] = val;
    }
    loaded_dc.push_back(std::move(dc));
  }

  uint8_t has_guard = 0;
  if (!ReadPod(in, &has_guard)) return false;
  if (has_guard != 0) {
    R765PreserveWindowGuard tmp_guard;
    if (!tmp_guard.LoadState(in)) return false;
    if (preserve_guard != nullptr) {
      *preserve_guard = std::move(tmp_guard);
    }
  }

  train_cohorts_ = std::move(loaded_tc);
  diag_cohorts_ = std::move(loaded_dc);
  last_batch_map_.clear();
  return true;
}

// --- WO-PERF-TIMING: the phase timer. ---------------------------------------
//
// Ordered exactly as the work order's §3.2 table. This array is the single
// source of truth for the order; the sidecar's field list is generated from
// it, so a reordering cannot silently mislabel a column.
const char* const kPhaseTimingNames[kPhaseTimingNumPhases] = {
    "tensor_pack_h2d", "diag_prepass",  "ppo_forward_loss", "backward",
    "grad_telemetry",  "grad_clip",     "optimizer_step",   "scalar_reads",
};

namespace {

// The CUDA event pool, REUSED across updates.
//
// PhaseTimer is a per-update local, so a pool owned by the timer would create
// and destroy every event on every update -- at the production shape that is
// ~2,100 cudaEventCreateWithFlags/cudaEventDestroy pairs per update, paid
// forever, for no benefit. Hoisting the pool here makes the reuse the earlier
// comment claimed but did not implement: events are created once, then
// re-recorded each update. Re-recording is well defined (cudaEventRecord
// overwrites) and safe here because Finalize reads every pair's elapsed time
// before the update returns, so no pair is ever re-recorded between its record
// and its read.
//
// thread_local, not a plain static: TrainPpoUpdate is called on one thread, but
// a pool shared across threads would be a data race waiting for the first
// caller that changes that.
std::vector<c10::Event>& PhaseEventPool() {
  static thread_local std::vector<c10::Event> pool;
  return pool;
}

}  // namespace

PhaseTimer::PhaseTimer(bool enabled, torch::Device device)
    : enabled_(enabled),
      cuda_(enabled && device.is_cuda()),
      device_(device) {
  for (int i = 0; i < kPhaseTimingNumPhases; ++i) open_event_[i] = kNoEvent;
  if (cuda_) impl_.emplace(device.type());
}

size_t PhaseTimer::AcquirePair() {
  auto& pool = PhaseEventPool();
  while (next_free_ + 1 >= pool.size()) {
    pool.emplace_back(device_.type(), c10::EventFlag::BACKEND_DEFAULT);
    pool.emplace_back(device_.type(), c10::EventFlag::BACKEND_DEFAULT);
  }
  const size_t idx = next_free_;
  next_free_ += 2;
  return idx;
}

void PhaseTimer::Begin(int phase) {
  if (!enabled_) return;
  host_begin_[phase] = std::chrono::steady_clock::now();
  if (!cuda_) return;
  const size_t idx = AcquirePair();
  open_event_[phase] = idx;
  PhaseEventPool()[idx].record(impl_->getStream(device_));
  ++events_recorded_;
}

void PhaseTimer::End(int phase) {
  if (!enabled_) return;
  if (cuda_) {
    const size_t idx = open_event_[phase];
    // An End without a matching Begin would otherwise index the pool blind.
    // Dropping the bracket is the safe failure: it lands in the reported
    // residual instead of fabricating an interval.
    if (idx == kNoEvent) return;
    PhaseEventPool()[idx + 1].record(impl_->getStream(device_));
    ++events_recorded_;
    pairs_.push_back({phase, idx});
    last_recorded_ = idx + 1;
    open_event_[phase] = kNoEvent;
  }
  host_s_[phase] += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - host_begin_[phase])
                        .count();
  ++bracket_count_;
}

void PhaseTimer::Finalize(PpoPhaseTimings* out) {
  if (!enabled_) return;
  out->enabled = true;
  out->cuda = cuda_;
  out->bracket_count = bracket_count_;
  out->events_recorded = events_recorded_;

  if (cuda_ && last_recorded_ != kNoEvent) {
    auto& pool = PhaseEventPool();
    // Every event was recorded on the stream returned by getStream() for this
    // device, and stream events complete in stream order, so waiting on the
    // LAST one is sufficient for all of them. One synchronise, never one per
    // phase -- a per-phase sync would serialise the pipeline and inflate the
    // very total this is trying to attribute (work order §3.3).
    const auto sync_begin = std::chrono::steady_clock::now();
    pool[last_recorded_].synchronize();
    out->sync_s = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - sync_begin)
                      .count();
    for (const auto& p : pairs_) {
      // elapsedTime is milliseconds on every backend that implements it.
      out->device_s[p.phase] +=
          pool[p.start].elapsedTime(pool[p.start + 1]) * 1e-3;
    }
  }

  for (int i = 0; i < kPhaseTimingNumPhases; ++i) {
    out->host_s[i] = host_s_[i];
    // On CPU there is no second clock to consult: steady_clock IS the
    // measurement, and the async hazard that motivates events does not arise.
    // Populating the field keeps the sidecar schema uniform across devices
    // instead of emitting nulls a consumer must special-case.
    if (!cuda_) out->device_s[i] = host_s_[i];
    out->total_attributed_s += out->device_s[i];
    out->total_host_attributed_s += out->host_s[i];
  }
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH
