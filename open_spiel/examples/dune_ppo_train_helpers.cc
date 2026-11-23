#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

std::set<std::vector<open_spiel::Action>> prohibited_histories;

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

// Copies BY NAME, not by position.
//
// PWO-5 makes the two models' module sets deliberately asymmetric: the training
// model carries the three auxiliary heads and the inference model does not
// (section 8.6 -- the heads must never be computed in a rollout/inference
// forward, and the cleanest way to guarantee that is for them not to exist
// there). The previous positional loop indexed `target_params[i]` for every `i`
// in the SOURCE's range, so a source with 6 extra tensors would have walked off
// the end of the target.
//
// CopyModelWeights is declared and defined inline in dune_network.h

void SyncModels(std::shared_ptr<SharedDunePolicyValueNetImpl> training_model,
                std::shared_ptr<SharedDunePolicyValueNetImpl> inference_model,
                std::shared_mutex* sync_mutex) {
  std::unique_lock<std::shared_mutex> lock(*sync_mutex);
  CopyModelWeights(training_model, inference_model);
}

void SyncDualModels(
    std::shared_ptr<SharedDunePolicyValueNetImpl> training_actor,
    std::shared_ptr<SharedDunePolicyValueNetImpl> inference_actor,
    std::shared_ptr<SharedDunePolicyValueNetImpl> training_critic,
    std::shared_ptr<SharedDunePolicyValueNetImpl> inference_critic,
    std::shared_mutex* sync_mutex) {
  std::unique_lock<std::shared_mutex> lock(*sync_mutex);
  CopyModelWeights(training_actor, inference_actor);
  CopyModelWeights(training_critic, inference_critic);
}


std::string HashNonValueParameters(const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::stringstream ss;
  torch::NoGradGuard no_grad;
  bool unfreeze_trunk = absl::GetFlag(FLAGS_unfreeze_trunk);
  for (auto& name_param : model->named_parameters()) {
    std::string name = name_param.key();
    bool should_hash = false;
    if (unfreeze_trunk) {
      if (name.rfind("policy_head", 0) == 0) {
        should_hash = true;
      }
    } else {
      if (name.find("value_head") == std::string::npos) {
        should_hash = true;
      }
    }
    if (should_hash) {
      auto tensor = name_param.value().contiguous().cpu().to(torch::kFloat32);
      float* data = static_cast<float*>(tensor.data_ptr());
      size_t num_el = tensor.numel();
      ss.write(reinterpret_cast<const char*>(data), num_el * sizeof(float));
    }
  }
  return open_spiel::ComputeStringSHA256(ss.str());
}

// Canonical in-memory identity for the read-only parity gate. Parameter and
// buffer names, shapes and FP32 bytes are included so equality before/after the
// diagnostic proves that neither collection nor replay mutated the model.
std::string HashAllModelState(
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::stringstream ss;
  torch::NoGradGuard no_grad;
  auto append = [&](const std::string& name, const torch::Tensor& source) {
    const torch::Tensor tensor =
        source.detach().contiguous().cpu().to(torch::kFloat32);
    ss << name << ':';
    for (int64_t dim : tensor.sizes()) ss << dim << ',';
    ss << ';';
    ss.write(reinterpret_cast<const char*>(tensor.data_ptr<float>()),
             tensor.numel() * sizeof(float));
  };
  for (const auto& item : model->named_parameters()) {
    append("P/" + item.key(), item.value());
  }
  for (const auto& item : model->named_buffers()) {
    append("B/" + item.key(), item.value());
  }
  return ComputeStringSHA256(ss.str());
}

// PWO-5 section 7.2 — THE ARCHITECTURE-VERSIONED CHECKPOINT MIGRATION.
//
// Loads trunk, policy head, value head "WITHOUT NUMERICAL CHANGE" from a
// checkpoint that may PREDATE the three auxiliary heads, and leaves the heads at
// their deterministic init (section 7.2) when the checkpoint does not carry
// them.
//
// The mechanism is the temp-model copy already proven in this file for the
// nonlinear-value-head case below, plus one probe. It relies on a documented
// property of torch: `Module::load(InputArchive&)` iterates the MODULE's own
// parameters/buffers/children and reads each by key, so
//
//   * loading a PRE-migration archive into a HEAD-LESS temp model succeeds, and
//   * a POST-migration archive's extra head entries are simply never visited.
//
// So the temp model reads both generations, and the only question is whether to
// then also pull the heads across -- which `try_read` on the head submodule
// answers without ever partially mutating `model`.
//
// Numerical exactness: the trunk/policy/value tensors are `copy_`d from the
// loaded temp model, which is a bitwise copy of the same dtype and shape. This
// is what makes section 7.3's bitwise legacy-inference parity gate passable.

double GetExpectedGroupLearningRate(size_t group_idx, size_t total_groups, double base_lr, double new_block_lr) {
  if (new_block_lr > 0.0 && total_groups >= 3 && group_idx == total_groups - 1) {
    return new_block_lr;
  }
  return base_lr;
}

void SetOptimizerLearningRate(torch::optim::Optimizer& optimizer, double lr, double new_block_lr) {
  const size_t total_groups = optimizer.param_groups().size();
  for (size_t g = 0; g < total_groups; ++g) {
    auto& options = static_cast<torch::optim::AdamWOptions&>(optimizer.param_groups()[g].options());
    double exp_lr = GetExpectedGroupLearningRate(g, total_groups, lr, new_block_lr);
    options.lr(exp_lr);
  }
}

std::string SummarizeParamNames(const std::vector<std::string>& names) {
  if (names.empty()) return "(none)";
  int input_count = 0;
  int value_count = 0;
  int policy_count = 0;
  int scorer_count = 0;
  std::map<int, int> res_counts;
  std::vector<std::string> other;

  for (const auto& name : names) {
    if (name.rfind("input_layer", 0) == 0) {
      ++input_count;
    } else if (name.rfind("policy_head", 0) == 0) {
      ++policy_count;
    } else if (name.rfind("value_head", 0) == 0) {
      ++value_count;
    } else if (name.rfind("semantic_scorer", 0) == 0) {
      ++scorer_count;
    } else if (name.rfind("res", 0) == 0) {
      size_t dot_pos = name.find('.');
      int blk_idx = 0;
      if (dot_pos != std::string::npos && dot_pos > 3) {
        blk_idx = std::atoi(name.substr(3, dot_pos - 3).c_str());
      }
      res_counts[blk_idx]++;
    } else {
      other.push_back(name);
    }
  }

  std::vector<std::string> parts;
  if (policy_count > 0) parts.push_back(absl::StrFormat("policy_head.* (%d)", policy_count));
  if (input_count > 0) parts.push_back(absl::StrFormat("input_layer.* (%d)", input_count));
  if (!res_counts.empty()) {
    int min_blk = res_counts.begin()->first;
    int max_blk = res_counts.rbegin()->first;
    int total_res = 0;
    for (const auto& p : res_counts) total_res += p.second;
    if (min_blk == max_blk) {
      parts.push_back(absl::StrFormat("res%d.* (%d)", min_blk, total_res));
    } else {
      parts.push_back(absl::StrFormat("res%d-res%d.* (%d)", min_blk, max_blk, total_res));
    }
  }
  if (value_count > 0) parts.push_back(absl::StrFormat("value_head.* (%d)", value_count));
  if (scorer_count > 0) parts.push_back(absl::StrFormat("semantic_scorer.* (%d)", scorer_count));
  for (const auto& o : other) parts.push_back(o);

  return absl::StrJoin(parts, ", ");
}

void PrintOptimizerGroupSummary(
    const torch::optim::Optimizer& optimizer,
    const std::shared_ptr<SharedDunePolicyValueNetImpl>& model) {
  std::unordered_map<const c10::TensorImpl*, std::string> tensor_to_name;
  for (const auto& pair : model->named_parameters()) {
    tensor_to_name[pair.value().unsafeGetTensorImpl()] = pair.key();
  }

  const size_t num_groups = optimizer.param_groups().size();
  std::cout << "[OPTIMIZER LOAD] Parameter groups configuration (" << num_groups << " groups):\n";
  for (size_t g = 0; g < num_groups; ++g) {
    const auto& group = optimizer.param_groups()[g];
    const auto& options = static_cast<const torch::optim::AdamWOptions&>(group.options());
    std::vector<std::string> param_names;
    for (const auto& p : group.params()) {
      auto it = tensor_to_name.find(p.unsafeGetTensorImpl());
      if (it != tensor_to_name.end()) {
        param_names.push_back(it->second);
      } else {
        param_names.push_back("unknown_param");
      }
    }
    std::string summary = SummarizeParamNames(param_names);
    std::cout << absl::StrFormat(
        "  Group %zu: lr=%.6e, weight_decay=%.6e, tensors=%zu, params=[%s]\n",
        g, options.lr(), options.weight_decay(), group.params().size(), summary);
  }
}

std::unique_ptr<torch::optim::AdamW> MakeOptimizer(
    std::shared_ptr<SharedDunePolicyValueNetImpl> model) {
  if (absl::GetFlag(FLAGS_train_value_only)) {
    std::vector<torch::Tensor> trainable_params;
    for (auto& param : model->parameters()) {
      if (param.requires_grad()) {
        trainable_params.push_back(param);
      }
    }
    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(trainable_params);
    auto optimizer = std::make_unique<torch::optim::AdamW>(
        groups, torch::optim::AdamWOptions(absl::GetFlag(FLAGS_learning_rate)).eps(1e-5));
    static_cast<torch::optim::AdamWOptions&>(
        optimizer->param_groups()[0].options())
        .weight_decay(absl::GetFlag(FLAGS_weight_decay));
    return optimizer;
  }

  return open_spiel::MakeDuneOptimizer(
      model,
      absl::GetFlag(FLAGS_learning_rate),
      absl::GetFlag(FLAGS_weight_decay),
      absl::GetFlag(FLAGS_policy_weight_decay),
      absl::GetFlag(FLAGS_new_block_lr),
      absl::GetFlag(FLAGS_new_block_first_index));
}

std::string GenerateUUID() {
  std::random_device rd;
  std::mt19937_64 generator(rd());
  uint64_t r1 = generator();
  uint64_t r2 = generator();
  // Set version 4 (random) and variant (RFC 4122)
  uint32_t a = (r1 >> 32);
  uint16_t b = (r1 >> 16) & 0xFFFF;
  uint16_t c = (r1 & 0x0FFF) | 0x4000;
  uint16_t d = ((r2 >> 48) & 0x3FFF) | 0x8000;
  uint64_t e = r2 & 0xFFFFFFFFFFFFULL;
  return absl::StrFormat("%08x-%04x-%04x-%04x-%012llx", a, b, c, d, e);
}

// Acceptance prior source this run will actually use: the flag when stated,
// otherwise the collector's library default. Unrecognized names are fatal —
// never silently fall back, since the whole point of the field is that the
// coverage contract is chosen explicitly (WO-20).
AcceptancePriorSource EffectiveAcceptancePriorSource() {
  const std::string name = absl::GetFlag(FLAGS_collector_acceptance_prior);
  if (name.empty()) return OnlineSearchConfig().acceptance_prior_source;
  AcceptancePriorSource source;
  if (!ParseAcceptancePriorSource(name, &source)) {
    SpielFatalError(absl::StrCat(
        "--collector_acceptance_prior=", name, " is not a known prior source. ",
        "Use '", AcceptancePriorSourceName(AcceptancePriorSource::kRawNetworkPrior),
        "' or '", AcceptancePriorSourceName(AcceptancePriorSource::kTreePrior),
        "'."));
  }
  return source;
}

std::string ComputeLegacyConfigFingerprint() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["terminal_reward_mode"] = absl::GetFlag(FLAGS_terminal_reward_mode);
  config_obj["round7_speed_bonus"] = absl::GetFlag(FLAGS_round7_speed_bonus);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  // Phase 18B online auxiliary-search collection (combined optimization).
  config_obj["online_search_collection"] = absl::GetFlag(FLAGS_online_search_collection);
  config_obj["auxiliary_games"] = absl::GetFlag(FLAGS_auxiliary_games);
  config_obj["auxiliary_search_seed_domain"] = static_cast<int64_t>(absl::GetFlag(FLAGS_auxiliary_search_seed_domain));
  config_obj["collector_dirichlet_epsilon"] = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
  config_obj["search_loss_coef"] = absl::GetFlag(FLAGS_search_loss_coef);
  config_obj["search_loss_warmup_update"] = absl::GetFlag(FLAGS_search_loss_warmup_update);
  config_obj["abort_grad_norm_ratio"] = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  config_obj["target_sharpen_exponent"] = absl::GetFlag(FLAGS_target_sharpen_exponent);
  config_obj["swordmaster_grant_fraction"] = absl::GetFlag(FLAGS_swordmaster_grant_fraction);
  config_obj["swordmaster_grant_round"] = absl::GetFlag(FLAGS_swordmaster_grant_round);
  // Leader teacher: pinned into the fingerprint so a resume cannot silently
  // change what the network is being taught at Leader nodes.
  //
  // Added CONDITIONALLY, and that is deliberate. An unconditional key would
  // change config_fingerprint for every existing run -- including runs whose
  // flags are entirely unchanged -- and every one of them would fail its resume
  // integrity check against a checkpoint written before this field existed.
  // With the flag off the object is byte-identical to before, so pre-Leader
  // runs resume untouched; with it on the fingerprint differs, which is exactly
  // the detection this pinning is for.
  if (absl::GetFlag(FLAGS_search_leader_draft)) {
    config_obj["search_leader_draft"] = true;
    config_obj["leader_draft_simulations"] =
        absl::GetFlag(FLAGS_leader_draft_simulations);
    config_obj["leader_mass_only_coverage"] =
        absl::GetFlag(FLAGS_leader_mass_only_coverage);
  }
  if (absl::GetFlag(FLAGS_family_atomics_penalty) != 0.0) {
    config_obj["family_atomics_penalty"] = absl::GetFlag(FLAGS_family_atomics_penalty);
  }
  if (absl::GetFlag(FLAGS_plot_intrigue_penalty) != 0.0) {
    config_obj["plot_intrigue_penalty"] = absl::GetFlag(FLAGS_plot_intrigue_penalty);
    config_obj["plot_intrigue_exemption_threshold"] = absl::GetFlag(FLAGS_plot_intrigue_exemption_threshold);
  }

  std::string json_str = open_spiel::json::ToString(config_obj);
  return open_spiel::ComputeStringSHA256(json_str);
}

json::Object BuildPrePrecisionConfigFingerprintObject() {
  open_spiel::json::Object config_obj;
  config_obj["game"] = absl::GetFlag(FLAGS_game);
  config_obj["ppo_minibatch_size"] = absl::GetFlag(FLAGS_ppo_minibatch_size);
  config_obj["ppo_update_epochs"] = absl::GetFlag(FLAGS_ppo_update_epochs);
  config_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
  config_obj["anneal_lr"] = absl::GetFlag(FLAGS_anneal_lr);
  config_obj["gamma"] = absl::GetFlag(FLAGS_gamma);
  config_obj["gae_lambda"] = absl::GetFlag(FLAGS_gae_lambda);
  config_obj["normalize_advantages"] = absl::GetFlag(FLAGS_normalize_advantages);
  config_obj["ppo_clip_epsilon"] = absl::GetFlag(FLAGS_ppo_clip_epsilon);
  config_obj["ppo_clip_value_loss"] = absl::GetFlag(FLAGS_ppo_clip_value_loss);
  config_obj["entropy_coef"] = absl::GetFlag(FLAGS_entropy_coef);
  config_obj["value_coef"] = absl::GetFlag(FLAGS_value_coef);
  config_obj["grad_clip_norm"] = absl::GetFlag(FLAGS_grad_clip_norm);
  config_obj["target_kl"] = absl::GetFlag(FLAGS_target_kl);
  config_obj["weight_decay"] = absl::GetFlag(FLAGS_weight_decay);
  config_obj["policy_weight_decay"] = absl::GetFlag(FLAGS_policy_weight_decay);
  config_obj["hidden_dim"] = absl::GetFlag(FLAGS_hidden_dim);
  config_obj["num_blocks"] = absl::GetFlag(FLAGS_num_blocks);
  config_obj["logit_cap"] = absl::GetFlag(FLAGS_logit_cap);
  config_obj["shaped_reward_weight"] = absl::GetFlag(FLAGS_shaped_reward_weight);
  config_obj["tleilaxu_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_breadcrumb_weight);
  config_obj["tleilaxu_level7_breadcrumb_weight"] = absl::GetFlag(FLAGS_tleilaxu_level7_breadcrumb_weight);
  config_obj["specimen_exchange_penalty"] = absl::GetFlag(FLAGS_specimen_exchange_penalty);
  config_obj["shaping_start_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_start_env_steps));
  config_obj["shaping_decay_env_steps"] = static_cast<int64_t>(absl::GetFlag(FLAGS_shaping_decay_env_steps));
  config_obj["reward_scale"] = absl::GetFlag(FLAGS_reward_scale);
  config_obj["terminal_reward_mode"] = absl::GetFlag(FLAGS_terminal_reward_mode);
  config_obj["round7_speed_bonus"] = absl::GetFlag(FLAGS_round7_speed_bonus);
  config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
  config_obj["pipeline"] = absl::GetFlag(FLAGS_pipeline);
  config_obj["rollout_games"] = absl::GetFlag(FLAGS_rollout_games);
  config_obj["rollout_transitions"] = absl::GetFlag(FLAGS_rollout_transitions);
  config_obj["search_lambda"] = absl::GetFlag(FLAGS_search_lambda);
  config_obj["search_minibatches_per_update"] = absl::GetFlag(FLAGS_search_minibatches_per_update);
  config_obj["search_minibatch_size"] = absl::GetFlag(FLAGS_search_minibatch_size);

  // Phase 18B online auxiliary-search collection (combined optimization).
  config_obj["online_search_collection"] = absl::GetFlag(FLAGS_online_search_collection);
  config_obj["auxiliary_games"] = absl::GetFlag(FLAGS_auxiliary_games);
  config_obj["auxiliary_search_seed_domain"] = static_cast<int64_t>(absl::GetFlag(FLAGS_auxiliary_search_seed_domain));
  config_obj["collector_dirichlet_epsilon"] = absl::GetFlag(FLAGS_collector_dirichlet_epsilon);
  config_obj["search_loss_coef"] = absl::GetFlag(FLAGS_search_loss_coef);
  config_obj["search_loss_warmup_update"] = absl::GetFlag(FLAGS_search_loss_warmup_update);
  config_obj["abort_grad_norm_ratio"] = absl::GetFlag(FLAGS_abort_grad_norm_ratio);
  config_obj["target_sharpen_exponent"] = absl::GetFlag(FLAGS_target_sharpen_exponent);
  config_obj["swordmaster_grant_fraction"] = absl::GetFlag(FLAGS_swordmaster_grant_fraction);
  config_obj["swordmaster_grant_round"] = absl::GetFlag(FLAGS_swordmaster_grant_round);
  // Leader teacher: pinned into the fingerprint so a resume cannot silently
  // change what the network is being taught at Leader nodes.
  //
  // Added CONDITIONALLY, and that is deliberate. An unconditional key would
  // change config_fingerprint for every existing run -- including runs whose
  // flags are entirely unchanged -- and every one of them would fail its resume
  // integrity check against a checkpoint written before this field existed.
  // With the flag off the object is byte-identical to before, so pre-Leader
  // runs resume untouched; with it on the fingerprint differs, which is exactly
  // the detection this pinning is for.
  if (absl::GetFlag(FLAGS_search_leader_draft)) {
    config_obj["search_leader_draft"] = true;
    config_obj["leader_draft_simulations"] =
        absl::GetFlag(FLAGS_leader_draft_simulations);
    config_obj["leader_mass_only_coverage"] =
        absl::GetFlag(FLAGS_leader_mass_only_coverage);
  }
  // Acceptance prior source is training semantics, not a launcher detail: it
  // decides which searches become CE targets. Fingerprinted so two runs that
  // measure coverage differently cannot stamp their examples with the same
  // provenance hash (WO-20). Deliberately NOT added to the legacy fingerprint
  // above — that one is a frozen field set whose job is to recognize older
  // manifests, so extending it would defeat its purpose.
  config_obj["collector_acceptance_prior"] =
      std::string(AcceptancePriorSourceName(EffectiveAcceptancePriorSource()));

  // New flags added for complete config fingerprint validation
  config_obj["deterministic"] = absl::GetFlag(FLAGS_deterministic);
  config_obj["deterministic_rollout_eval"] = absl::GetFlag(FLAGS_deterministic_rollout_eval);
  // Critic-remediation settings are training semantics, not launcher details.
  // Persist them in the fingerprint so a partially completed value-only run
  // cannot be resumed with a different set of trainable parameters or data.
  config_obj["train_value_only"] = absl::GetFlag(FLAGS_train_value_only);
  config_obj["unfreeze_trunk"] = absl::GetFlag(FLAGS_unfreeze_trunk);
  config_obj["nonlinear_value_head"] =
      absl::GetFlag(FLAGS_nonlinear_value_head);
  config_obj["sample_counterfactual_states"] =
      absl::GetFlag(FLAGS_sample_counterfactual_states);
  config_obj["counterfactual_samples_per_game"] =
      absl::GetFlag(FLAGS_counterfactual_samples_per_game);
  config_obj["counterfactual_replay_weight"] =
      absl::GetFlag(FLAGS_counterfactual_replay_weight);
  config_obj["policy_kl_anchor_coeff"] =
      absl::GetFlag(FLAGS_policy_kl_anchor_coeff);

  if (absl::GetFlag(FLAGS_single_learner_training)) {
    config_obj["single_learner_training"] = true;
    config_obj["training_opponent_pool"] = absl::GetFlag(FLAGS_training_opponent_pool);
  }

  if (absl::GetFlag(FLAGS_pool_game_fraction) > 0.0 &&
      absl::GetFlag(FLAGS_pool_opponents_per_game) > 0) {
    config_obj["pool_game_fraction"] = absl::GetFlag(FLAGS_pool_game_fraction);
    config_obj["pool_opponents_per_game"] = absl::GetFlag(FLAGS_pool_opponents_per_game);
    config_obj["training_opponent_pool"] = absl::GetFlag(FLAGS_training_opponent_pool);
  }

  const auto resolved_deck_pool_semantics =
      dune_imperium::ParseDeckPoolSemanticsOrDie(absl::GetFlag(FLAGS_deck_pool_semantics));
  if (absl::GetFlag(FLAGS_market_appendix_mode) == "full_public_information_v5") {
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::MarketAppendixMode::kFullPublicInformationV5,
        resolved_deck_pool_semantics);
    config_obj["market_appendix_mode"] = std::string("full_public_information_v5");
    config_obj["observation_dim"] = schema.size;
    config_obj["feature_schema_sha256"] = std::string(schema.sha256);
  } else if (absl::GetFlag(FLAGS_market_appendix_mode) == "full_public_information_v4") {
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::MarketAppendixMode::kFullPublicInformationV4,
        resolved_deck_pool_semantics);
    config_obj["market_appendix_mode"] = std::string("full_public_information_v4");
    config_obj["observation_dim"] = schema.size;
    config_obj["feature_schema_sha256"] = std::string(schema.sha256);
  } else if (absl::GetFlag(FLAGS_market_appendix_mode) == "full_public_information_v3") {
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::MarketAppendixMode::kFullPublicInformationV3,
        resolved_deck_pool_semantics);
    config_obj["market_appendix_mode"] = std::string("full_public_information_v3");
    config_obj["observation_dim"] = schema.size;
    config_obj["feature_schema_sha256"] = std::string(schema.sha256);
  } else if (absl::GetFlag(FLAGS_market_appendix_mode) == "ordered_card_slots_v2") {
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::MarketAppendixMode::kOrderedCardSlotsV2,
        resolved_deck_pool_semantics);
    config_obj["market_appendix_mode"] = std::string("ordered_card_slots_v2");
    config_obj["observation_dim"] = schema.size;
    config_obj["feature_schema_sha256"] = std::string(schema.sha256);
  }

  if (resolved_deck_pool_semantics == dune_imperium::DeckPoolSemantics::kFixed) {
    config_obj["deck_pool_semantics"] = std::string("fixed");
  }

  if (absl::GetFlag(FLAGS_enable_semantic_scorer)) {
    config_obj["enable_semantic_scorer"] = true;
    config_obj["semantic_descriptor_schema"] = absl::GetFlag(FLAGS_semantic_descriptor_schema);
  }
  if (absl::GetFlag(FLAGS_family_atomics_penalty) != 0.0) {
    config_obj["family_atomics_penalty"] = absl::GetFlag(FLAGS_family_atomics_penalty);
  }
  if (absl::GetFlag(FLAGS_plot_intrigue_penalty) != 0.0) {
    config_obj["plot_intrigue_penalty"] = absl::GetFlag(FLAGS_plot_intrigue_penalty);
    config_obj["plot_intrigue_exemption_threshold"] = absl::GetFlag(FLAGS_plot_intrigue_exemption_threshold);
  }
  if (absl::GetFlag(FLAGS_ema_magnet_kl_coef) > 0.0) {
    config_obj["ema_magnet_kl_coef"] = absl::GetFlag(FLAGS_ema_magnet_kl_coef);
  }
  if (absl::GetFlag(FLAGS_v5_appendix_lr_mult) > 0.0) {
    config_obj["v5_appendix_lr_mult"] = absl::GetFlag(FLAGS_v5_appendix_lr_mult);
  }
  if (absl::GetFlag(FLAGS_win_speed_bonus_per_round) != 0.0) {
    config_obj["win_speed_bonus_per_round"] = absl::GetFlag(FLAGS_win_speed_bonus_per_round);
  }

  return config_obj;
}

std::string ComputeCurrentPrePrecisionConfigFingerprint() {
  return open_spiel::ComputePrePrecisionConfigFingerprint(
      BuildPrePrecisionConfigFingerprintObject());
}

std::string ComputeConfigFingerprint() {
  auto obj = BuildPrePrecisionConfigFingerprintObject();
  return open_spiel::ComputePrecisionConfigFingerprint(
      obj,
      absl::GetFlag(FLAGS_rollout_amp),
      absl::GetFlag(FLAGS_allow_tf32));
}

// ---------------------------------------------------------------------------
std::string GetSearchLabelFingerprint(const std::string& search_label_dir) {
  if (search_label_dir.empty()) {
    return "";
  }
  std::filesystem::path manifest_path = std::filesystem::path(search_label_dir) / "manifest.json";
  if (!std::filesystem::exists(manifest_path)) {
    std::cout << "Warning: search_label_dir is specified but manifest.json not found at " << manifest_path << "\n";
    return "";
  }
  try {
    std::ifstream ifs(manifest_path);
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    auto val_opt = open_spiel::json::FromString(content);
    if (val_opt.has_value() && val_opt->IsObject()) {
      const auto& obj = val_opt->GetObject();
      auto it = obj.find("search_label_fingerprint");
      if (it != obj.end() && it->second.IsString()) {
        return it->second.GetString();
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading search label manifest at " << manifest_path << ": " << e.what() << "\n";
  }
  return "";
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
