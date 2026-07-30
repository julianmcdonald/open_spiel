#ifndef OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_LABEL_BUFFER_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_LABEL_BUFFER_H_

// The offline search-label pack loader, and PWO-5 amendment 1 ruling 5's
// explicit per-file ROLES.
//
// EXTRACTED FROM dune_ppo_train.cc SO THERE IS EXACTLY ONE DEFINITION.
//
// Ruling 5's whole content is that the loader must honour whole-game roles
// instead of a per-row hash, and the registration makes "no held-out row may
// receive a training gradient" a checkable property rather than a claim. A test
// that re-stated the loader would prove nothing about the loader the trainer
// links -- this is the same "one shared definition, called by both the shipped
// path and the test" pattern the Memocorders fix used for
// dune_terminal_vp_report.h, and for the same reason: the previous defect was
// three reimplementations of one rule drifting apart.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

#include "dune_network.h"
#include "dune_ppo_training_utils.h"
#include "dune_sha256.h"

namespace open_spiel {

struct SearchLabel {
  std::vector<float> state;
  int32_t num_legal_actions;
  std::vector<std::pair<int64_t, float>> teacher_probs;
  std::vector<std::pair<int64_t, float>> ppo_probs;
  float teacher_kl;
  int32_t num_covered_actions;
  float eta;
  uint8_t eta_capped;
  uint8_t player_id;
};

// PWO-5 amendment 1 ruling 5: explicit per-file roles.
//
// The legacy loader's built-in split is IsValidationLabel -- a PER-ROW FNV-1a
// hash mod 11 applied to every row of every file. On the PWO-5 pack it diverts
// 1,890 of 20,582 rows (9.18%) into the loader's own validation bucket, which
// the distillation loss never trains on: a registered exposure of 20,582 rows
// and 14.93 expected passes silently became 18,692 and 16.43.
//
// The fix is NOT to tune around the hash. A manifest file entry may carry an
// explicit `role`, and a file that carries one assigns EVERY one of its rows to
// that role with IsValidationLabel not consulted at all. A file carrying NO
// role keeps the legacy %11 behaviour exactly, so no committed result changes
// and no other consumer of this loader is affected.
enum class SearchLabelRole {
  kLegacyHash,   // no role declared -> IsValidationLabel per row (legacy)
  kTrain,        // every row is gradient-eligible
  kValidation,   // every row is held out from all gradients
};

struct SearchLabelFileEntry {
  std::string filename;
  std::string sha256;
  SearchLabelRole role = SearchLabelRole::kLegacyHash;
  bool role_declared = false;
};

class SearchLabelBuffer {
 public:
  void SetExpectedDimensions(int64_t obs_size, int64_t action_dim) {
    expected_obs_size_ = obs_size;
    expected_action_dim_ = action_dim;
  }

  // True once a role-aware manifest has been loaded. Callers use it to decide
  // whether the realized-count contract applies.
  bool role_aware() const { return role_aware_; }

  void LoadFromDirectory(const std::string& dir) {
    if (dir.empty() || !std::filesystem::exists(dir)) return;

    // Verify manifest.json
    std::filesystem::path manifest_path = std::filesystem::path(dir) / "manifest.json";
    if (!std::filesystem::exists(manifest_path)) {
      SpielFatalError("SearchLabelBuffer: manifest.json not found in " + dir);
    }

    std::ifstream ifs(manifest_path.string());
    if (!ifs) {
      SpielFatalError("SearchLabelBuffer: Failed to open manifest.json in " + dir);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    auto val_opt = open_spiel::json::FromString(content);
    if (!val_opt) {
      SpielFatalError("SearchLabelBuffer: Failed to parse manifest.json in " + dir);
    }

    const auto& manifest_obj = val_opt->GetObject();

    // Verify schema_version
    auto schema_it = manifest_obj.find("schema_version");
    if (schema_it == manifest_obj.end() || static_cast<int>(schema_it->second.GetInt()) != 2) {
      SpielFatalError("SearchLabelBuffer: Unsupported manifest schema version in " + dir + " (expected 2)");
    }

    // Verify training_label_count & validation_label_count
    auto train_cnt_it = manifest_obj.find("training_label_count");
    auto val_cnt_it = manifest_obj.find("validation_label_count");
    if (train_cnt_it == manifest_obj.end() || val_cnt_it == manifest_obj.end()) {
      SpielFatalError("SearchLabelBuffer: Missing training/validation counts in manifest.json");
    }

    int64_t training_count = train_cnt_it->second.GetInt();
    int64_t validation_count = val_cnt_it->second.GetInt();
    if (training_count < 8192 || validation_count < 1024) {
      SpielFatalError("SearchLabelBuffer: Manifest training count (" + std::to_string(training_count) +
                     ") or validation count (" + std::to_string(validation_count) + ") is insufficient.");
    }

    // Verify files & SHA-256 hashes, and read the optional per-file role.
    auto files_it = manifest_obj.find("files");
    if (files_it == manifest_obj.end()) {
      SpielFatalError("SearchLabelBuffer: Missing 'files' list in manifest.json");
    }
    const auto& files_arr = files_it->second.GetArray();
    if (files_arr.empty()) {
      SpielFatalError("SearchLabelBuffer: 'files' list in manifest.json is empty");
    }
    std::vector<SearchLabelFileEntry> entries;
    std::set<std::string> seen_filenames;
    bool any_role = false;
    for (const auto& f_val : files_arr) {
      const auto& f_obj = f_val.GetObject();
      auto fn_it = f_obj.find("filename");
      auto hash_it = f_obj.find("sha256");
      if (fn_it == f_obj.end() || hash_it == f_obj.end()) {
        SpielFatalError("SearchLabelBuffer: Missing filename or sha256 in file entry");
      }

      SearchLabelFileEntry entry;
      entry.filename = fn_it->second.GetString();
      entry.sha256 = hash_it->second.GetString();

      // Reject anything that is not a plain `.bin` basename in this directory.
      // A manifest that can name `../x` or a non-.bin file is a manifest that
      // can pull an unaudited artifact into a gradient-eligible role.
      if (entry.filename.empty() ||
          entry.filename.find('/') != std::string::npos ||
          entry.filename.find('\\') != std::string::npos ||
          entry.filename.find("..") != std::string::npos) {
        SpielFatalError("SearchLabelBuffer: invalid filename in manifest file "
                        "entry: '" + entry.filename + "' (must be a plain "
                        "basename inside the label directory)");
      }
      if (entry.filename.size() < 4 ||
          entry.filename.compare(entry.filename.size() - 4, 4, ".bin") != 0) {
        SpielFatalError("SearchLabelBuffer: manifest file entry '" +
                        entry.filename + "' is not a .bin label pack");
      }
      if (!seen_filenames.insert(entry.filename).second) {
        SpielFatalError("SearchLabelBuffer: duplicate manifest file entry '" +
                        entry.filename + "'");
      }

      auto role_it = f_obj.find("role");
      if (role_it != f_obj.end()) {
        if (!role_it->second.IsString()) {
          SpielFatalError("SearchLabelBuffer: file entry '" + entry.filename +
                          "' has a non-string role");
        }
        const std::string role_str = role_it->second.GetString();
        if (role_str == "train") {
          entry.role = SearchLabelRole::kTrain;
        } else if (role_str == "validation") {
          entry.role = SearchLabelRole::kValidation;
        } else {
          SpielFatalError("SearchLabelBuffer: unknown role '" + role_str +
                          "' on file entry '" + entry.filename +
                          "' (the only accepted roles are \"train\" and "
                          "\"validation\")");
        }
        entry.role_declared = true;
        any_role = true;
      }

      std::filesystem::path bin_path = std::filesystem::path(dir) / entry.filename;
      if (!std::filesystem::exists(bin_path)) {
        SpielFatalError("SearchLabelBuffer: Label file " + entry.filename + " listed in manifest does not exist.");
      }

      std::string actual_sha256 = open_spiel::ComputeFileSHA256(bin_path.string());
      if (actual_sha256 != entry.sha256) {
        SpielFatalError("SearchLabelBuffer: Hash mismatch for file " + entry.filename +
                       ": expected=" + entry.sha256 + " actual=" + actual_sha256);
      }
      entries.push_back(entry);
    }
    role_aware_ = any_role;

    // Reconstruct semantic object for fingerprint verification.
    //
    // For a ROLELESS legacy manifest the object is byte-identical to what it
    // has always been, so every committed legacy fingerprint still verifies.
    // A ROLE-AWARE manifest additionally binds the (filename, sha256, role)
    // mapping into the fingerprint: without that, the roles would be the one
    // part of the contract nothing authenticates, and an edited manifest could
    // silently move the held-out pack into the training bucket.
    open_spiel::json::Object semantic_obj;
    semantic_obj["schema_version"] = manifest_obj.at("schema_version");
    semantic_obj["base_seed"] = manifest_obj.at("base_seed");
    semantic_obj["model_checkpoint_sha256"] = manifest_obj.at("model_checkpoint_sha256");
    semantic_obj["effective_search_config"] = manifest_obj.at("effective_search_config");
    semantic_obj["architecture"] = manifest_obj.at("architecture");
    semantic_obj["training_label_count"] = manifest_obj.at("training_label_count");
    semantic_obj["validation_label_count"] = manifest_obj.at("validation_label_count");
    if (role_aware_) {
      semantic_obj["file_roles"] =
          open_spiel::json::Value(CanonicalFileRoles(entries));
    }

    std::string semantic_json = open_spiel::json::ToString(semantic_obj);
    std::string expected_fingerprint = open_spiel::ComputeStringSHA256(semantic_json);

    auto fp_it = manifest_obj.find("search_label_fingerprint");
    if (fp_it == manifest_obj.end() || fp_it->second.GetString() != expected_fingerprint) {
      SpielFatalError("SearchLabelBuffer: Manifest search_label_fingerprint mismatch!");
    }

    std::cout << "SearchLabelBuffer: manifest.json verified successfully. Fingerprint: "
              << expected_fingerprint << "\n";

    if (!role_aware_) {
      // Unchanged legacy path: rescan the directory and split every row by %11.
      LoadNewFiles(dir);
      return;
    }

    // Role-aware path. Load EXACTLY the manifest's files, in the manifest's
    // order, each with its declared role -- never a directory rescan, which
    // would drop the role mapping on the floor and reload the files as
    // roleless. And no unlisted `.bin` may sit in a role-aware directory: it
    // would be an unhashed, unroled pack one edit away from entering a
    // gradient-eligible dataset.
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (const auto& entry : entries) {
        const std::string path =
            (std::filesystem::path(dir) / entry.filename).string();
        LoadFile(path, entry.role);
        loaded_files_.insert(path);
      }
      for (const auto& dir_entry : std::filesystem::directory_iterator(dir)) {
        if (dir_entry.path().extension() != ".bin") continue;
        if (seen_filenames.count(dir_entry.path().filename().string()) == 0) {
          SpielFatalError(
              "SearchLabelBuffer: '" + dir_entry.path().filename().string() +
              "' is a .bin file in a role-aware label directory but is not "
              "listed in manifest.json. Unlisted packs are rejected rather "
              "than ignored: an ignored pack looks identical to a pack that "
              "was silently loaded.");
        }
      }
    }

    // Ruling 5 item 3: the realized per-role counts must EQUAL the manifest's
    // declared counts. The previous code checked the declared counts against
    // the floors and never against reality, so a manifest declaring 20,582 and
    // a loader materializing 18,692 disagreed silently.
    if (static_cast<int64_t>(labels_.size()) != training_count ||
        static_cast<int64_t>(validation_labels_.size()) != validation_count) {
      SpielFatalError(
          "SearchLabelBuffer: realized role counts do not match the manifest. "
          "declared train=" + std::to_string(training_count) +
          " validation=" + std::to_string(validation_count) +
          "; realized train=" + std::to_string(labels_.size()) +
          " validation=" + std::to_string(validation_labels_.size()));
    }
    std::cout << "SearchLabelBuffer: role-aware manifest realized "
              << labels_.size() << " train / " << validation_labels_.size()
              << " validation rows, matching the declared counts.\n";
  }

  // The canonical, order-sensitive serialization of the file/role mapping that
  // enters a role-aware manifest's fingerprint. Order-sensitive because the
  // load order is the manifest's order.
  static std::string CanonicalFileRoles(
      const std::vector<SearchLabelFileEntry>& entries) {
    std::string s;
    for (const auto& e : entries) {
      s += e.filename;
      s += ':';
      s += e.sha256;
      s += ':';
      switch (e.role) {
        case SearchLabelRole::kTrain: s += "train"; break;
        case SearchLabelRole::kValidation: s += "validation"; break;
        case SearchLabelRole::kLegacyHash: s += "legacy"; break;
      }
      s += ';';
    }
    return s;
  }

  void LoadNewFiles(const std::string& dir) {
    if (dir.empty() || !std::filesystem::exists(dir)) return;
    // A role-aware dataset is closed at LoadFromDirectory time: its membership
    // is exactly the manifest's list. Rescanning would be the precise defect
    // ruling 5 exists to remove, so it is refused rather than skipped.
    if (role_aware_) {
      SpielFatalError(
          "SearchLabelBuffer::LoadNewFiles called on a role-aware label "
          "directory. Role-aware datasets are closed at load time; a rescan "
          "would reload the manifest's files without their roles.");
    }
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
      if (entry.path().extension() == ".bin") {
        paths.push_back(entry.path().string());
      }
    }
    std::sort(paths.begin(), paths.end());
    for (const auto& path_str : paths) {
      if (loaded_files_.find(path_str) == loaded_files_.end()) {
        LoadFile(path_str, SearchLabelRole::kLegacyHash);
        loaded_files_.insert(path_str);
      }
    }
  }

  std::vector<SearchLabel> Sample(int n, std::mt19937* rng) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<SearchLabel> batch;
    if (labels_.empty()) return batch;
    batch.reserve(n);
    std::uniform_int_distribution<size_t> dist(0, labels_.size() - 1);
    for (int i = 0; i < n; ++i) {
      batch.push_back(labels_[dist(*rng)]);
    }
    return batch;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return labels_.size();
  }

  double ComputeValidationKL(const std::shared_ptr<open_spiel::SharedDunePolicyValueNetImpl>& model,
                             const torch::Device& device, float logit_cap) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (validation_labels_.empty()) return 0.0;

    torch::NoGradGuard no_grad;
    const size_t batch_size = 512;
    double kl_sum = 0.0;
    size_t total_count = 0;

    for (size_t i = 0; i < validation_labels_.size(); i += batch_size) {
      size_t current_batch_size = std::min(batch_size, validation_labels_.size() - i);

      torch::Tensor states_cpu = torch::zeros({static_cast<int64_t>(current_batch_size), expected_obs_size_}, torch::kFloat);
      torch::Tensor masks_cpu = torch::zeros({static_cast<int64_t>(current_batch_size), expected_action_dim_}, torch::kBool);
      torch::Tensor teacher_probs_cpu = torch::zeros({static_cast<int64_t>(current_batch_size), expected_action_dim_}, torch::kFloat);

      float* states_ptr = states_cpu.data_ptr<float>();
      bool* masks_ptr = masks_cpu.data_ptr<bool>();
      float* teacher_ptr = teacher_probs_cpu.data_ptr<float>();

      for (size_t j = 0; j < current_batch_size; ++j) {
        const auto& label = validation_labels_[i + j];
        std::copy(label.state.begin(), label.state.end(), states_ptr + j * expected_obs_size_);
        for (const auto& ap : label.teacher_probs) {
          int action_id = ap.first;
          float prob = ap.second;
          masks_ptr[j * expected_action_dim_ + action_id] = true;
          teacher_ptr[j * expected_action_dim_ + action_id] = prob;
        }
      }

      torch::Tensor states = states_cpu.to(device);
      torch::Tensor masks = masks_cpu.to(device);
      torch::Tensor teacher_probs = teacher_probs_cpu.to(device);

      auto outputs = model->forward(states);
      torch::Tensor logits = CenterAndCapLogitsTensor(outputs.logits, masks, logit_cap);
      torch::Tensor masked_logits = logits.masked_fill(masks.logical_not(), -1e9f);
      torch::Tensor log_probs = torch::log_softmax(masked_logits, -1);

      torch::Tensor log_teacher = torch::log(teacher_probs.clamp_min(1e-12f));
      torch::Tensor kl_loss = teacher_probs * (log_teacher - log_probs);
      torch::Tensor mean_kl = kl_loss.sum(-1);

      kl_sum += mean_kl.sum().item<double>();
      total_count += current_batch_size;
    }

    return total_count > 0 ? (kl_sum / total_count) : 0.0;
  }

  size_t ValidationSize() const {
    std::lock_guard<std::mutex> lock(mu_);
    return validation_labels_.size();
  }

  // Read-only row access, for the ruling-5 role tests ONLY. They exist so
  // "no held-out row can receive a training gradient" is proved EXHAUSTIVELY
  // -- by inspecting every row the sampler can reach -- and not only by
  // drawing from the sampler and hoping the draw was representative. Nothing
  // in the training path calls these.
  const SearchLabel& TrainingLabelForTest(size_t i) const {
    std::lock_guard<std::mutex> lock(mu_);
    return labels_.at(i);
  }
  const SearchLabel& ValidationLabelForTest(size_t i) const {
    std::lock_guard<std::mutex> lock(mu_);
    return validation_labels_.at(i);
  }

 private:
  // Caller holds mu_.
  void LoadFile(const std::string& path, SearchLabelRole role) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      std::cerr << "SearchLabelBuffer: Failed to open file: " << path << "\n";
      return;
    }

    uint32_t magic = 0;
    uint32_t schema = 0;
    int32_t obs_size = 0;
    int32_t action_dim = 0;
    int32_t max_simulations = 0;
    float utility_divisor = 0;
    float puct_c = 0;
    float target_teacher_kl = 0;
    int32_t min_visits = 0;
    int32_t min_coverage = 0;
    float blueprint_temp = 0;
    uint64_t fingerprint = 0;
    uint32_t reserved = 0;

    in.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x4c545344) {
      std::cerr << "SearchLabelBuffer: Invalid magic in " << path << "\n";
      return;
    }
    in.read(reinterpret_cast<char*>(&schema), 4);
    if (schema != 1 && schema != 2) {
      std::cerr << "SearchLabelBuffer: Unsupported schema version " << schema
                << " in " << path << " (expected 1 or 2)\n";
      return;
    }
    in.read(reinterpret_cast<char*>(&obs_size), 4);
    in.read(reinterpret_cast<char*>(&action_dim), 4);
    if (expected_obs_size_ > 0 && obs_size != expected_obs_size_) {
      std::cerr << "SearchLabelBuffer: obs_size mismatch in " << path
                << ": file=" << obs_size << " expected=" << expected_obs_size_ << "\n";
      return;
    }
    if (expected_action_dim_ > 0 && action_dim != expected_action_dim_) {
      std::cerr << "SearchLabelBuffer: action_dim mismatch in " << path
                << ": file=" << action_dim << " expected=" << expected_action_dim_ << "\n";
      return;
    }
    in.read(reinterpret_cast<char*>(&max_simulations), 4);
    in.read(reinterpret_cast<char*>(&utility_divisor), 4);
    if (utility_divisor != 4.0f) {
      std::cerr << "SearchLabelBuffer: utility_divisor mismatch in " << path
                << ": file=" << utility_divisor << " expected=4.0\n";
      return;
    }
    in.read(reinterpret_cast<char*>(&puct_c), 4);
    in.read(reinterpret_cast<char*>(&target_teacher_kl), 4);
    in.read(reinterpret_cast<char*>(&min_visits), 4);
    in.read(reinterpret_cast<char*>(&min_coverage), 4);
    in.read(reinterpret_cast<char*>(&blueprint_temp), 4);
    in.read(reinterpret_cast<char*>(&fingerprint), 8);
    if (!has_fingerprint_) {
      expected_fingerprint_ = fingerprint;
      has_fingerprint_ = true;
    } else if (fingerprint != expected_fingerprint_) {
      std::cerr << "Warning: mixed label fingerprints in " << path
                << ": file fingerprint=0x" << std::hex << fingerprint
                << " expected=0x" << expected_fingerprint_ << std::dec << "\n";
    }
    in.read(reinterpret_cast<char*>(&reserved), 4);

    int labels_before = labels_.size();
    int val_before = validation_labels_.size();
    while (in.peek() != EOF) {
      SearchLabel label;
      label.state.resize(obs_size);
      in.read(reinterpret_cast<char*>(label.state.data()), obs_size * sizeof(float));
      if (!in) break;

      in.read(reinterpret_cast<char*>(&label.num_legal_actions), sizeof(int32_t));
      if (label.num_legal_actions <= 0 || label.num_legal_actions > action_dim) {
        std::cerr << "SearchLabelBuffer: Invalid num_legal_actions " << label.num_legal_actions
                  << " in " << path << "\n";
        break;
      }
      label.teacher_probs.resize(label.num_legal_actions);
      label.ppo_probs.resize(label.num_legal_actions);

      bool valid = true;
      for (int32_t i = 0; i < label.num_legal_actions; ++i) {
        int32_t action_id = 0;
        float t_prob = 0.0f;
        float p_prob = 0.0f;
        in.read(reinterpret_cast<char*>(&action_id), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&t_prob), sizeof(float));
        in.read(reinterpret_cast<char*>(&p_prob), sizeof(float));
        if (action_id < 0 || action_id >= action_dim ||
            !std::isfinite(t_prob) || !std::isfinite(p_prob)) {
          valid = false;
          break;
        }
        label.teacher_probs[i] = {action_id, t_prob};
        label.ppo_probs[i] = {action_id, p_prob};
      }
      if (!valid || !in) break;
      in.read(reinterpret_cast<char*>(&label.teacher_kl), sizeof(float));
      in.read(reinterpret_cast<char*>(&label.num_covered_actions), sizeof(int32_t));
      in.read(reinterpret_cast<char*>(&label.eta), sizeof(float));
      in.read(reinterpret_cast<char*>(&label.eta_capped), sizeof(uint8_t));

      uint8_t pid = 0;
      uint8_t padding[2];
      in.read(reinterpret_cast<char*>(&pid), sizeof(uint8_t));
      in.read(reinterpret_cast<char*>(padding), 2);
      if (!in) break;
      label.player_id = pid;

      // A declared role assigns EVERY row of the file to that role, and
      // IsValidationLabel is not consulted for it at all. A roleless file
      // keeps the legacy per-row hash.
      bool is_validation;
      switch (role) {
        case SearchLabelRole::kTrain:
          is_validation = false;
          break;
        case SearchLabelRole::kValidation:
          is_validation = true;
          break;
        case SearchLabelRole::kLegacyHash:
        default: {
          std::vector<int32_t> legal_actions;
          for (const auto& ap : label.teacher_probs) {
            legal_actions.push_back(static_cast<int32_t>(ap.first));
          }
          is_validation =
              IsValidationLabel(label.state, legal_actions, label.player_id);
          break;
        }
      }

      if (is_validation) {
        validation_labels_.push_back(std::move(label));
      } else {
        labels_.push_back(std::move(label));
      }
    }
    int labels_loaded = labels_.size() - labels_before;
    int val_loaded = validation_labels_.size() - val_before;
    std::cout << "SearchLabelBuffer: Loaded " << labels_loaded
              << " train, " << val_loaded << " val labels from " << path << "\n";
  }

  int64_t expected_obs_size_ = 0;
  int64_t expected_action_dim_ = 0;
  uint64_t expected_fingerprint_ = 0;
  bool has_fingerprint_ = false;
  bool role_aware_ = false;
  std::vector<SearchLabel> labels_;
  std::vector<SearchLabel> validation_labels_;
  std::set<std::string> loaded_files_;
  mutable std::mutex mu_;
};

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_SEARCH_LABEL_BUFFER_H_
