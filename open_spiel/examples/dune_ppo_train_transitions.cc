#include "dune_ppo_train_internal.h"

namespace open_spiel::ppo_train_internal {

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

std::string ValidateAndComputeRewardTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_reward_transition)) {
    return "";
  }
  std::ifstream mfs(manifest_path);
  if (!mfs) {
    SpielFatalError("Could not open manifest file at: " + manifest_path);
  }
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  auto get_double_or_default = [&](const std::string& k, double def) -> double {
    auto it = manifest_obj.find(k);
    if (it != manifest_obj.end() && it->second.IsDouble()) return it->second.GetDouble();
    return def;
  };
  auto get_int_or_default = [&](const std::string& k, int64_t def) -> int64_t {
    auto it = manifest_obj.find(k);
    if (it != manifest_obj.end() && it->second.IsInt()) return it->second.GetInt();
    return def;
  };

  double source_specimen = get_double_or_default("specimen_exchange_penalty", 0.0);
  double source_family_atomics = get_double_or_default("family_atomics_penalty", 0.0);
  double source_plot = get_double_or_default("plot_intrigue_penalty", 0.0);
  int64_t source_plot_thresh = get_int_or_default("plot_intrigue_exemption_threshold", 0);
  double source_win_speed = get_double_or_default("win_speed_bonus_per_round", 0.0);

  int64_t source_rollout_games = absl::GetFlag(FLAGS_reward_transition_source_rollout_games);
  if (source_rollout_games <= 0) {
    source_rollout_games = get_int_or_default("rollout_games", 0);
  }

  // Reconstruct source pre-precision config from current flags with source penalties substituted
  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  source_config_obj["specimen_exchange_penalty"] = source_specimen;
  if (source_family_atomics != 0.0) {
    source_config_obj["family_atomics_penalty"] = source_family_atomics;
  } else {
    source_config_obj.erase("family_atomics_penalty");
  }
  if (source_plot != 0.0) {
    source_config_obj["plot_intrigue_penalty"] = source_plot;
    source_config_obj["plot_intrigue_exemption_threshold"] = source_plot_thresh;
  } else {
    source_config_obj.erase("plot_intrigue_penalty");
    source_config_obj.erase("plot_intrigue_exemption_threshold");
  }
  if (source_win_speed != 0.0) {
    source_config_obj["win_speed_bonus_per_round"] = source_win_speed;
  } else {
    source_config_obj.erase("win_speed_bonus_per_round");
  }
  auto it_r7_speed = manifest_obj.find("round7_speed_bonus");
  if (it_r7_speed != manifest_obj.end() && it_r7_speed->second.IsDouble()) {
    source_config_obj["round7_speed_bonus"] = it_r7_speed->second.GetDouble();
  }
  auto it_term_mode = manifest_obj.find("terminal_reward_mode");
  if (it_term_mode != manifest_obj.end() && it_term_mode->second.IsString()) {
    source_config_obj["terminal_reward_mode"] = it_term_mode->second.GetString();
  }
  if (source_rollout_games > 0) {
    source_config_obj["rollout_games"] = source_rollout_games;
  }
  if (absl::GetFlag(FLAGS_allow_pool_transition)) {
    auto pgf_it = manifest_obj.find("pool_game_fraction");
    auto pop_it = manifest_obj.find("pool_opponents_per_game");
    auto top_it = manifest_obj.find("training_opponent_pool");
    double src_pgf = (pgf_it != manifest_obj.end() && pgf_it->second.IsDouble()) ? pgf_it->second.GetDouble() : 0.0;
    int64_t src_pop = (pop_it != manifest_obj.end() && pop_it->second.IsInt()) ? pop_it->second.GetInt() : 0;
    std::string src_top = (top_it != manifest_obj.end() && top_it->second.IsString()) ? top_it->second.GetString() : "";
    if (src_pgf > 0.0 && src_pop > 0) {
      source_config_obj["pool_game_fraction"] = src_pgf;
      source_config_obj["pool_opponents_per_game"] = src_pop;
      source_config_obj["training_opponent_pool"] = src_top;
    } else {
      source_config_obj.erase("pool_game_fraction");
      source_config_obj.erase("pool_opponents_per_game");
      source_config_obj.erase("training_opponent_pool");
    }
  }

  std::string computed_source_fp = open_spiel::ComputePrecisionConfigFingerprint(
      source_config_obj,
      absl::GetFlag(FLAGS_rollout_amp),
      absl::GetFlag(FLAGS_allow_tf32));

  auto fp_it = manifest_obj.find("config_fingerprint");
  if (it_r7_speed == manifest_obj.end() &&
      fp_it != manifest_obj.end() && fp_it->second.IsString() &&
      fp_it->second.GetString() != computed_source_fp &&
      absl::GetFlag(FLAGS_round7_speed_bonus) != 0.0) {
    auto fallback_obj = source_config_obj;
    fallback_obj["round7_speed_bonus"] = 0.0;
    std::string fallback_fp = open_spiel::ComputePrecisionConfigFingerprint(
        fallback_obj, absl::GetFlag(FLAGS_rollout_amp), absl::GetFlag(FLAGS_allow_tf32));
    if (fallback_fp == fp_it->second.GetString()) {
      source_config_obj = std::move(fallback_obj);
      computed_source_fp = std::move(fallback_fp);
    }
  }

  std::string expected_source_fp = absl::GetFlag(FLAGS_reward_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Reward transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }

  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Reward transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }

  std::cout << absl::StrFormat(
      "[reward-transition] Validated intentional reward continuation transition:\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  Specimen penalty:   %.4f -> %.4f\n"
      "  Plot penalty:       %.4f -> %.4f\n"
      "  Atomics penalty:    %.4f (unchanged)\n"
      "  Win speed/round:    %.4f -> %.4f\n"
      "  Rollout games:      %d -> %d\n",
      computed_source_fp, current_config_fingerprint,
      source_specimen, absl::GetFlag(FLAGS_specimen_exchange_penalty),
      source_plot, absl::GetFlag(FLAGS_plot_intrigue_penalty),
      source_family_atomics,
      source_win_speed, absl::GetFlag(FLAGS_win_speed_bonus_per_round),
      source_rollout_games > 0 ? source_rollout_games : absl::GetFlag(FLAGS_rollout_games),
      absl::GetFlag(FLAGS_rollout_games)) << std::endl;

  return computed_source_fp;
}

std::string ValidateAndComputeGaeTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_gae_transition)) {
    return "";
  }
  std::ifstream mfs(manifest_path);
  if (!mfs) {
    SpielFatalError("Could not open manifest file at: " + manifest_path);
  }
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  double source_lambda = absl::GetFlag(FLAGS_gae_transition_source_lambda);

  // Reconstruct source pre-precision config from current flags with source lambda substituted
  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  source_config_obj["gae_lambda"] = source_lambda;

  std::string computed_source_fp = open_spiel::ComputePrecisionConfigFingerprint(
      source_config_obj,
      absl::GetFlag(FLAGS_rollout_amp),
      absl::GetFlag(FLAGS_allow_tf32));

  std::string expected_source_fp = absl::GetFlag(FLAGS_gae_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "GAE transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }

  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "GAE transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_gae_transition permits ONLY gae_lambda to differ from the parent manifest. All other settings must match exactly.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }

  std::cout << absl::StrFormat(
      "[gae-transition] Validated intentional GAE lambda continuation transition:\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  GAE lambda:         %.4f -> %.4f\n",
      computed_source_fp, current_config_fingerprint,
      source_lambda, absl::GetFlag(FLAGS_gae_lambda)) << std::endl;

  return computed_source_fp;
}

std::string ValidateAndComputeLearningRateTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_learning_rate_transition)) {
    return "";
  }
  std::ifstream mfs(manifest_path);
  if (!mfs) {
    SpielFatalError("Could not open manifest file at: " + manifest_path);
  }
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  double source_lr = absl::GetFlag(FLAGS_learning_rate_transition_source_lr);
  double source_v5_lr_mult = 0.0;
  auto it_v5_mult = manifest_obj.find("v5_appendix_lr_mult");
  if (it_v5_mult != manifest_obj.end() && it_v5_mult->second.IsDouble()) {
    source_v5_lr_mult = it_v5_mult->second.GetDouble();
  }

  // Reconstruct source pre-precision config from current flags with source learning rate substituted
  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  source_config_obj["learning_rate"] = source_lr;
  if (source_v5_lr_mult > 0.0) {
    source_config_obj["v5_appendix_lr_mult"] = source_v5_lr_mult;
  } else {
    source_config_obj.erase("v5_appendix_lr_mult");
  }

  std::string computed_source_fp = open_spiel::ComputePrecisionConfigFingerprint(
      source_config_obj,
      absl::GetFlag(FLAGS_rollout_amp),
      absl::GetFlag(FLAGS_allow_tf32));

  auto fp_it = manifest_obj.find("config_fingerprint");
  auto* lr_src_flag = absl::FindCommandLineFlag("learning_rate_transition_source_lr");
  const bool lr_src_explicit =
      (lr_src_flag != nullptr && lr_src_flag->CurrentValue() != lr_src_flag->DefaultValue());
  if (!lr_src_explicit &&
      fp_it != manifest_obj.end() && fp_it->second.IsString() &&
      fp_it->second.GetString() != computed_source_fp &&
      absl::GetFlag(FLAGS_learning_rate) != source_lr) {
    auto same_lr_obj = source_config_obj;
    same_lr_obj["learning_rate"] = absl::GetFlag(FLAGS_learning_rate);
    std::string same_lr_fp = open_spiel::ComputePrecisionConfigFingerprint(
        same_lr_obj, absl::GetFlag(FLAGS_rollout_amp), absl::GetFlag(FLAGS_allow_tf32));
    if (same_lr_fp == fp_it->second.GetString()) {
      source_lr = absl::GetFlag(FLAGS_learning_rate);
      source_config_obj = std::move(same_lr_obj);
      computed_source_fp = std::move(same_lr_fp);
    }
  }

  std::string expected_source_fp = absl::GetFlag(FLAGS_learning_rate_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Learning rate transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }

  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Learning rate transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_learning_rate_transition permits ONLY learning_rate / v5_appendix_lr_mult to differ from the parent manifest. All other settings must match exactly.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }

  std::cout << absl::StrFormat(
      "[lr-transition] Validated intentional learning rate continuation transition:\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  Learning rate:      %.4e -> %.4e\n"
      "  V5 appendix mult:   %.4f -> %.4f\n",
      computed_source_fp, current_config_fingerprint,
      source_lr, absl::GetFlag(FLAGS_learning_rate),
      source_v5_lr_mult, absl::GetFlag(FLAGS_v5_appendix_lr_mult)) << std::endl;

  return computed_source_fp;
}

std::string ValidateAndComputeTargetKlTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_target_kl_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();
  const double source_target_kl =
      absl::GetFlag(FLAGS_target_kl_transition_source_target_kl);
  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  source_config_obj["target_kl"] = source_target_kl;
  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, absl::GetFlag(FLAGS_rollout_amp),
          absl::GetFlag(FLAGS_allow_tf32));
  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_target_kl_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "target_kl source fingerprint mismatch: expected %s, computed %s",
        expected_source_fp, computed_source_fp));
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "target_kl transition requires a source manifest matching all settings except target_kl; stored=%s computed=%s",
        fp_it != manifest_obj.end() && fp_it->second.IsString()
            ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }
  std::cout << absl::StrFormat(
      "[target_kl-transition] %.4f -> %.4f; source fingerprint %s",
      source_target_kl, absl::GetFlag(FLAGS_target_kl), computed_source_fp)
            << std::endl;
  return computed_source_fp;
}

std::string ValidateAndComputeSingleLearnerTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_single_learner_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  int64_t source_rollout_games = absl::GetFlag(FLAGS_single_learner_transition_source_rollout_games);
  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  source_config_obj.erase("single_learner_training");
  source_config_obj.erase("training_opponent_pool");
  source_config_obj["rollout_games"] = source_rollout_games;

  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, absl::GetFlag(FLAGS_rollout_amp),
          absl::GetFlag(FLAGS_allow_tf32));
  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_single_learner_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Single-learner transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Single-learner transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_single_learner_transition permits single_learner_training and rollout_games to differ from parent manifest.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }
  std::cout << absl::StrFormat(
      "[single-learner-transition] Validated intentional single learner continuation transition:\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  Rollout games:      %d -> %d\n",
      computed_source_fp, current_config_fingerprint,
      source_rollout_games, absl::GetFlag(FLAGS_rollout_games)) << std::endl;
  return computed_source_fp;
}

std::string ValidateAndComputeEmaMagnetTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_ema_magnet_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();
  const double source_coef = absl::GetFlag(FLAGS_ema_magnet_transition_source_coef);
  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  if (source_coef > 0.0) {
    source_config_obj["ema_magnet_kl_coef"] = source_coef;
  } else {
    source_config_obj.erase("ema_magnet_kl_coef");
  }
  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, absl::GetFlag(FLAGS_rollout_amp),
          absl::GetFlag(FLAGS_allow_tf32));
  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_ema_magnet_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "EMA magnet transition source fingerprint mismatch: expected %s, computed %s",
        expected_source_fp, computed_source_fp));
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "EMA magnet transition requires a source manifest matching all settings except ema_magnet_kl_coef; stored=%s computed=%s",
        fp_it != manifest_obj.end() && fp_it->second.IsString()
            ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }
  std::cout << absl::StrFormat(
      "[ema_magnet-transition] %.4f -> %.4f; source fingerprint %s",
      source_coef, absl::GetFlag(FLAGS_ema_magnet_kl_coef), computed_source_fp)
            << std::endl;
  return computed_source_fp;
}

std::string ValidateAndComputePoolTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_pool_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();

  auto pgf_it = manifest_obj.find("pool_game_fraction");
  auto pop_it = manifest_obj.find("pool_opponents_per_game");
  auto top_it = manifest_obj.find("training_opponent_pool");

  double source_pgf = (pgf_it != manifest_obj.end() && pgf_it->second.IsDouble()) ? pgf_it->second.GetDouble() : 0.0;
  int64_t source_pop = (pop_it != manifest_obj.end() && pop_it->second.IsInt()) ? pop_it->second.GetInt() : 0;
  std::string source_top = (top_it != manifest_obj.end() && top_it->second.IsString()) ? top_it->second.GetString() : "";

  if (source_pgf > 0.0 && source_pop > 0) {
    source_config_obj["pool_game_fraction"] = source_pgf;
    source_config_obj["pool_opponents_per_game"] = source_pop;
    source_config_obj["training_opponent_pool"] = source_top;
  } else {
    source_config_obj.erase("pool_game_fraction");
    source_config_obj.erase("pool_opponents_per_game");
    source_config_obj.erase("training_opponent_pool");
  }
  if (absl::GetFlag(FLAGS_allow_reward_transition)) {
    auto get_dbl = [&](const std::string& k, double def) -> double {
      auto it = manifest_obj.find(k);
      return (it != manifest_obj.end() && it->second.IsDouble()) ? it->second.GetDouble() : def;
    };
    auto get_i64 = [&](const std::string& k, int64_t def) -> int64_t {
      auto it = manifest_obj.find(k);
      return (it != manifest_obj.end() && it->second.IsInt()) ? it->second.GetInt() : def;
    };
    source_config_obj["specimen_exchange_penalty"] = get_dbl("specimen_exchange_penalty", 0.0);
    double src_fa = get_dbl("family_atomics_penalty", 0.0);
    if (src_fa != 0.0) source_config_obj["family_atomics_penalty"] = src_fa;
    else source_config_obj.erase("family_atomics_penalty");
    double src_pl = get_dbl("plot_intrigue_penalty", 0.0);
    if (src_pl != 0.0) {
      source_config_obj["plot_intrigue_penalty"] = src_pl;
      source_config_obj["plot_intrigue_exemption_threshold"] = get_i64("plot_intrigue_exemption_threshold", 0);
    } else {
      source_config_obj.erase("plot_intrigue_penalty");
      source_config_obj.erase("plot_intrigue_exemption_threshold");
    }
    double src_ws = get_dbl("win_speed_bonus_per_round", 0.0);
    if (src_ws != 0.0) source_config_obj["win_speed_bonus_per_round"] = src_ws;
    else source_config_obj.erase("win_speed_bonus_per_round");
    source_config_obj["round7_speed_bonus"] = get_dbl("round7_speed_bonus", 0.0);
    int64_t src_rg = absl::GetFlag(FLAGS_reward_transition_source_rollout_games);
    if (src_rg <= 0) src_rg = get_i64("rollout_games", 0);
    if (src_rg > 0) source_config_obj["rollout_games"] = src_rg;
  }

  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, absl::GetFlag(FLAGS_rollout_amp),
          absl::GetFlag(FLAGS_allow_tf32));
  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_pool_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Pool transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Pool transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_pool_transition permits pool settings to differ from parent manifest.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }
  std::cout << absl::StrFormat(
      "[pool-transition] Validated intentional opponent pool continuation transition:\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  Pool settings: (frac=%.2f, opps=%lld, pool='%s') -> (frac=%.2f, opps=%d, pool='%s')\n",
      computed_source_fp, current_config_fingerprint,
      source_pgf, static_cast<long long>(source_pop), source_top,
      absl::GetFlag(FLAGS_pool_game_fraction), absl::GetFlag(FLAGS_pool_opponents_per_game),
      absl::GetFlag(FLAGS_training_opponent_pool)) << std::endl;
  return computed_source_fp;
}


std::string ValidateAndComputeMarketAppendixTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_market_appendix_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();

  std::string source_mode = absl::GetFlag(FLAGS_market_appendix_transition_source_mode);
  if (source_mode.empty()) {
    auto mode_it = manifest_obj.find("market_appendix_mode");
    if (mode_it != manifest_obj.end() && mode_it->second.IsString()) {
      source_mode = mode_it->second.GetString();
    }
  }
  if (source_mode.empty()) {
    source_mode = "none";
  }

  dune_imperium::DeckPoolSemantics source_sem = dune_imperium::DeckPoolSemantics::kLegacy;
  auto sem_it = manifest_obj.find("deck_pool_semantics");
  if (sem_it != manifest_obj.end() && sem_it->second.IsString()) {
    source_sem = dune_imperium::ParseDeckPoolSemanticsOrDie(sem_it->second.GetString());
  }

  if (source_mode == "none") {
    source_config_obj.erase("market_appendix_mode");
    source_config_obj.erase("observation_dim");
    source_config_obj.erase("feature_schema_sha256");
  } else {
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::ParseMarketAppendixMode(source_mode), source_sem);
    source_config_obj["market_appendix_mode"] = source_mode;
    source_config_obj["observation_dim"] = schema.size;
    source_config_obj["feature_schema_sha256"] = std::string(schema.sha256);
  }
  if (source_sem == dune_imperium::DeckPoolSemantics::kFixed) {
    source_config_obj["deck_pool_semantics"] = std::string("fixed");
  } else {
    source_config_obj.erase("deck_pool_semantics");
  }

  std::string source_semantic_schema;
  const bool also_semantic_transition =
      absl::GetFlag(FLAGS_allow_semantic_schema_transition) &&
      absl::GetFlag(FLAGS_enable_semantic_scorer);
  if (also_semantic_transition) {
    source_semantic_schema = "semantic_action_v3";
    auto schema_it = manifest_obj.find("semantic_descriptor_schema");
    if (schema_it != manifest_obj.end() && schema_it->second.IsString()) {
      source_semantic_schema = schema_it->second.GetString();
    }
    if (source_semantic_schema.empty() || source_semantic_schema == "none") {
      source_config_obj.erase("semantic_descriptor_schema");
    } else {
      source_config_obj["semantic_descriptor_schema"] = source_semantic_schema;
    }
  }

  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, absl::GetFlag(FLAGS_rollout_amp),
          absl::GetFlag(FLAGS_allow_tf32));
  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_market_appendix_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Market appendix transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }
  if (also_semantic_transition) {
    const std::string expected_sem_fp =
        absl::GetFlag(FLAGS_semantic_schema_transition_source_fingerprint);
    if (!expected_sem_fp.empty() && computed_source_fp != expected_sem_fp) {
      SpielFatalError(absl::StrFormat(
          "Semantic schema transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
          expected_sem_fp, computed_source_fp));
    }
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Market appendix transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_market_appendix_transition permits market_appendix_mode to differ from parent manifest.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }
  std::cout << absl::StrFormat(
      "[market_appendix-transition] Validated intentional market appendix transition:\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  Market appendix mode: '%s' -> '%s'\n",
      computed_source_fp, current_config_fingerprint,
      source_mode, absl::GetFlag(FLAGS_market_appendix_mode)) << std::endl;
  if (also_semantic_transition) {
    std::cout << absl::StrFormat(
        "[semantic_schema-transition] Validated simultaneous semantic schema transition:\n"
        "  Source fingerprint: %s\n"
        "  Target fingerprint: %s\n"
        "  Semantic schema:    '%s' -> '%s'\n",
        computed_source_fp, current_config_fingerprint,
        source_semantic_schema, absl::GetFlag(FLAGS_semantic_descriptor_schema)) << std::endl;
  }
  return computed_source_fp;
}

std::string ValidateAndComputeSemanticSchemaTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_semantic_schema_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  std::string source_schema = "semantic_action_v3";
  auto it = manifest_obj.find("semantic_descriptor_schema");
  if (it != manifest_obj.end() && it->second.IsString()) {
    source_schema = it->second.GetString();
  }

  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  if (source_schema.empty() || source_schema == "none") {
    source_config_obj.erase("semantic_descriptor_schema");
  } else {
    source_config_obj["semantic_descriptor_schema"] = source_schema;
  }

  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, absl::GetFlag(FLAGS_rollout_amp),
          absl::GetFlag(FLAGS_allow_tf32));
  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_semantic_schema_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Semantic schema transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "Semantic schema transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_semantic_schema_transition permits semantic_descriptor_schema to differ from parent manifest.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }
  std::cout << absl::StrFormat(
      "[semantic_schema-transition] Validated intentional semantic schema transition:\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  Semantic schema:    '%s' -> '%s'\n",
      computed_source_fp, current_config_fingerprint,
      source_schema, absl::GetFlag(FLAGS_semantic_descriptor_schema)) << std::endl;
  return computed_source_fp;
}

std::string ValidateAndComputeDeckPoolSemanticsTransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_deck_pool_semantics_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  open_spiel::ValidateResumeDeckPoolSemanticsOrDie(
      manifest_obj, absl::GetFlag(FLAGS_deck_pool_semantics), "resume");

  dune_imperium::DeckPoolSemantics source_sem = dune_imperium::DeckPoolSemantics::kLegacy;
  auto sem_it = manifest_obj.find("deck_pool_semantics");
  if (sem_it != manifest_obj.end() && sem_it->second.IsString()) {
    source_sem = dune_imperium::ParseDeckPoolSemanticsOrDie(sem_it->second.GetString());
  }

  std::string source_mode = "";
  auto mode_it = manifest_obj.find("market_appendix_mode");
  if (mode_it != manifest_obj.end() && mode_it->second.IsString()) {
    source_mode = mode_it->second.GetString();
  } else {
    source_mode = absl::GetFlag(FLAGS_market_appendix_mode);
  }
  if (source_mode.empty()) {
    source_mode = "none";
  }

  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();

  if (source_mode == "full_public_information_v5" ||
      source_mode == "full_public_information_v4" ||
      source_mode == "full_public_information_v3" ||
      source_mode == "ordered_card_slots_v2") {
    const auto schema = dune_imperium::GetInformationStateSchema(
        dune_imperium::ParseMarketAppendixMode(source_mode), source_sem);
    source_config_obj["market_appendix_mode"] = source_mode;
    source_config_obj["observation_dim"] = schema.size;
    source_config_obj["feature_schema_sha256"] = std::string(schema.sha256);
  } else if (source_mode == "none") {
    source_config_obj.erase("market_appendix_mode");
    source_config_obj.erase("observation_dim");
    source_config_obj.erase("feature_schema_sha256");
  }

  if (source_sem == dune_imperium::DeckPoolSemantics::kFixed) {
    source_config_obj["deck_pool_semantics"] = std::string("fixed");
  } else {
    source_config_obj.erase("deck_pool_semantics");
  }

  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, absl::GetFlag(FLAGS_rollout_amp),
          absl::GetFlag(FLAGS_allow_tf32));
  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_deck_pool_semantics_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "deck_pool_semantics transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "deck_pool_semantics transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_deck_pool_semantics_transition permits deck_pool_semantics to differ from parent manifest.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }

  int64_t source_global_update = -1;
  auto gu_it = manifest_obj.find("global_update");
  if (gu_it != manifest_obj.end() && gu_it->second.IsInt()) {
    source_global_update = gu_it->second.GetInt();
  }

  std::cout << absl::StrFormat(
      "[deck_pool_semantics-transition] Validated intentional deck_pool_semantics transition:\n"
      "  Source update:       %lld\n"
      "  Source fingerprint:  %s\n"
      "  Target fingerprint:  %s\n"
      "  Deck pool semantics: '%s' -> '%s'\n",
      static_cast<long long>(source_global_update),
      computed_source_fp, current_config_fingerprint,
      dune_imperium::DeckPoolSemanticsToString(source_sem),
      absl::GetFlag(FLAGS_deck_pool_semantics)) << std::endl;
  return computed_source_fp;
}

std::string ValidateAndComputeTf32TransitionFingerprint(
    const std::string& manifest_path,
    const std::string& current_config_fingerprint) {
  if (!absl::GetFlag(FLAGS_allow_tf32_transition)) return "";
  std::ifstream mfs(manifest_path);
  if (!mfs) SpielFatalError("Could not open manifest file at: " + manifest_path);
  std::string mcontent((std::istreambuf_iterator<char>(mfs)),
                      std::istreambuf_iterator<char>());
  auto val_opt = open_spiel::json::FromString(mcontent);
  if (!val_opt.has_value() || !val_opt->IsObject()) {
    SpielFatalError("Malformed manifest JSON at: " + manifest_path);
  }
  const auto& manifest_obj = val_opt->GetObject();

  bool source_rollout_amp = absl::GetFlag(FLAGS_rollout_amp);
  auto rollout_it = manifest_obj.find("rollout_amp");
  if (rollout_it != manifest_obj.end() && rollout_it->second.IsBool()) {
    source_rollout_amp = rollout_it->second.GetBool();
  }

  bool source_allow_tf32 = !absl::GetFlag(FLAGS_allow_tf32);
  auto tf32_it = manifest_obj.find("allow_tf32");
  if (tf32_it != manifest_obj.end() && tf32_it->second.IsBool()) {
    source_allow_tf32 = tf32_it->second.GetBool();
  }

  auto source_config_obj = BuildPrePrecisionConfigFingerprintObject();
  bool source_train_amp = absl::GetFlag(FLAGS_train_amp);
  auto train_amp_it = manifest_obj.find("train_amp");
  if (train_amp_it != manifest_obj.end() && train_amp_it->second.IsBool()) {
    source_train_amp = train_amp_it->second.GetBool();
    source_config_obj["train_amp"] = source_train_amp;
  } else {
    // If manifest doesn't have "train_amp" explicitly stored as a manifest root field:
    // Check if manifest_obj["config_fingerprint"] matches fp with current train_amp.
    std::string stored_fp = "";
    auto fp_it = manifest_obj.find("config_fingerprint");
    if (fp_it != manifest_obj.end() && fp_it->second.IsString()) {
      stored_fp = fp_it->second.GetString();
    }
    const std::string fp_same_train_amp = open_spiel::ComputePrecisionConfigFingerprint(
        source_config_obj, source_rollout_amp, source_allow_tf32);
    if (!stored_fp.empty() && stored_fp != fp_same_train_amp) {
      source_config_obj["train_amp"] = !absl::GetFlag(FLAGS_train_amp);
      const std::string fp_flipped = open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj, source_rollout_amp, source_allow_tf32);
      if (stored_fp == fp_flipped) {
        source_train_amp = !absl::GetFlag(FLAGS_train_amp);
      } else {
        source_config_obj["train_amp"] = absl::GetFlag(FLAGS_train_amp);
      }
    }
  }

  if (source_allow_tf32 == absl::GetFlag(FLAGS_allow_tf32) &&
      source_rollout_amp == absl::GetFlag(FLAGS_rollout_amp) &&
      source_train_amp == absl::GetFlag(FLAGS_train_amp)) {
    SpielFatalError(absl::StrFormat(
        "allow_tf32/precision transition was requested, but source manifest precision settings "
        "(rollout_amp=%s, allow_tf32=%s, train_amp=%s) match current settings.",
        source_rollout_amp ? "true" : "false",
        source_allow_tf32 ? "true" : "false",
        source_train_amp ? "true" : "false"));
  }

  const std::string computed_source_fp =
      open_spiel::ComputePrecisionConfigFingerprint(
          source_config_obj,
          source_rollout_amp,
          source_allow_tf32);

  const std::string expected_source_fp =
      absl::GetFlag(FLAGS_tf32_transition_source_fingerprint);
  if (!expected_source_fp.empty() && computed_source_fp != expected_source_fp) {
    SpielFatalError(absl::StrFormat(
        "allow_tf32/precision transition source fingerprint mismatch.\n  Expected by flag: %s\n  Computed from source settings: %s",
        expected_source_fp, computed_source_fp));
  }
  auto fp_it = manifest_obj.find("config_fingerprint");
  if (fp_it == manifest_obj.end() || !fp_it->second.IsString() ||
      fp_it->second.GetString() != computed_source_fp) {
    SpielFatalError(absl::StrFormat(
        "allow_tf32/precision transition source manifest fingerprint mismatch.\n  Manifest stored: %s\n  Computed from source settings: %s\n"
        "  Note: allow_tf32_transition permits ONLY precision settings (allow_tf32, rollout_amp, train_amp) to differ from the parent manifest. All other settings must match exactly.",
        (fp_it != manifest_obj.end() && fp_it->second.IsString()) ? fp_it->second.GetString() : "<missing>",
        computed_source_fp));
  }

  int64_t source_global_update = -1;
  auto gu_it = manifest_obj.find("global_update");
  if (gu_it != manifest_obj.end() && gu_it->second.IsInt()) {
    source_global_update = gu_it->second.GetInt();
  }

  std::cout << absl::StrFormat(
      "[allow_tf32-transition] Validated intentional allow_tf32 continuation transition:\n"
      "  Source update:      %lld\n"
      "  Source fingerprint: %s\n"
      "  Target fingerprint: %s\n"
      "  allow_tf32:         %s -> %s\n"
      "  rollout_amp:        %s -> %s\n"
      "  train_amp:          %s -> %s\n",
      static_cast<long long>(source_global_update),
      computed_source_fp, current_config_fingerprint,
      source_allow_tf32 ? "true" : "false",
      absl::GetFlag(FLAGS_allow_tf32) ? "true" : "false",
      source_rollout_amp ? "true" : "false",
      absl::GetFlag(FLAGS_rollout_amp) ? "true" : "false",
      source_train_amp ? "true" : "false",
      absl::GetFlag(FLAGS_train_amp) ? "true" : "false") << std::endl;
  return computed_source_fp;
}


#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel::ppo_train_internal
