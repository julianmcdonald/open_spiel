add_executable(benchmark_game benchmark_game.cc ${OPEN_SPIEL_OBJECTS})
add_test(benchmark_game_test benchmark_game --game=tic_tac_toe --sims=100 --attempts=2)

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_library(dune_search STATIC
    dune_puct_is_mcts.cc
    dune_puct_is_mcts_sim_helpers.cc
    dune_puct_is_mcts_sim.cc
    dune_search_routing.cc
    dune_search_session.cc
    dune_search_session_mcts.cc
    dune_search_session_adaptive_init.cc
    dune_search_session_adaptive_search.cc
    dune_search_session_adaptive_rollout.cc
    dune_search_session_adaptive_direct.cc)
  target_link_libraries(dune_search PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(dune_search PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(dune_search games open_spiel_core)

  add_library(dune_collector_obj OBJECT dune_online_search_collector.cc)
  target_link_libraries(dune_collector_obj PUBLIC dune_search ${TORCH_LIBRARIES})
  target_include_directories(dune_collector_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(dune_collector_obj dune_search games open_spiel_core)

  add_library(dune_ppo_utils_obj OBJECT
    dune_ppo_training_utils.cc
    dune_ppo_training_utils_manifest.cc
    dune_ppo_training_utils_checkpoint.cc
    dune_ppo_training_utils_load.cc
    dune_ppo_training_utils_state.cc
    dune_ppo_training_utils_diagnostics.cc
    dune_ppo_training_utils_separate.cc
    dune_ppo_training_utils_aux_eval.cc
    dune_ppo_training_utils_aux_phase.cc
    dune_ppo_training_utils_update.cc
    dune_ppo_training_utils_update_step.cc
    dune_ppo_training_utils_update_epoch_aux.cc
    dune_ppo_training_utils_update_epoch.cc
    dune_ppo_training_utils_update_finish.cc
    dune_ppo_training_flags.cc)
  target_link_libraries(dune_ppo_utils_obj PUBLIC dune_search ${TORCH_LIBRARIES})
  target_include_directories(dune_ppo_utils_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  add_dependencies(dune_ppo_utils_obj dune_search games open_spiel_core)

  add_executable(dune_ppo_train
    dune_ppo_train.cc
    dune_ppo_train_helpers.cc
    dune_ppo_train_transitions.cc
    dune_ppo_train_checkpoint.cc
    dune_ppo_train_rollout.cc
    dune_ppo_train_parity_replay.cc
    dune_ppo_train_parity_write.cc
    dune_ppo_train_gates.cc
    dune_ppo_train_sim.cc
    dune_ppo_train_sim_step.cc
    dune_ppo_train_sim_tracking.cc
    dune_ppo_train_startup.cc
    dune_ppo_train_resume.cc
    dune_ppo_train_bootstrap.cc
    dune_ppo_train_setup.cc
    dune_ppo_train_loop_update.cc
    dune_ppo_train_loop_aux.cc
    dune_ppo_train_loop_post.cc
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_ppo_train dune_search ${TORCH_LIBRARIES})

  add_executable(dune_evaluator dune_evaluator.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_evaluator ${TORCH_LIBRARIES})

  add_executable(dune_evaluator_test dune_evaluator_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_evaluator_test ${TORCH_LIBRARIES})

  add_executable(dune_puct_is_mcts_test dune_puct_is_mcts_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_puct_is_mcts_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_puct_is_mcts_test COMMAND dune_puct_is_mcts_test)

  # WO-1 Phase 4: teacher/collector acceptance parity on a shared fixture.
  add_executable(dune_acceptance_parity_test
    dune_acceptance_parity_test.cc $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_acceptance_parity_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_acceptance_parity_test COMMAND dune_acceptance_parity_test)

  add_executable(dune_online_search_collector_test
    dune_online_search_collector_test.cc $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_online_search_collector_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_online_search_collector_test COMMAND dune_online_search_collector_test)

  add_executable(dune_collector_smoke
    dune_collector_smoke.cc $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_collector_smoke dune_search ${TORCH_LIBRARIES})

  # Rung 4A: the two-model search controller. The class under test is a pure
  # composition of two algorithms::Evaluator instances, so the test needs no
  # torch model at all -- it drives two counting mocks and proves that priors
  # and leaf values leave through different doors. That independence is the
  # whole safety property of the hybrid arm, and a fused PriorAndEvaluate would
  # silently route the candidate's value head into search without it.
  add_executable(dune_split_evaluator_test
    dune_split_evaluator_test.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_split_evaluator_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_split_evaluator_test COMMAND dune_split_evaluator_test)

  add_executable(dune_split_parity_audit
    dune_split_parity_audit.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_split_parity_audit dune_search ${TORCH_LIBRARIES})

  add_executable(dune_eval_1000 dune_eval_1000.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_eval_1000 ${TORCH_LIBRARIES})

  add_executable(dune_population_eval
    dune_population_eval.cc
    dune_population_eval_setup.cc
    dune_population_eval_games.cc
    dune_population_eval_report.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_population_eval dune_search ${TORCH_LIBRARIES})

  add_executable(dune_search_benchmark
    dune_search_benchmark.cc
    dune_search_benchmark_setup.cc
    dune_search_benchmark_runner.cc
    dune_search_benchmark_jsonl.cc
    dune_search_benchmark_report.cc
    dune_search_benchmark_corpus.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_search_benchmark dune_search ${TORCH_LIBRARIES})

  add_executable(dune_stage_a_probe dune_stage_a_probe.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_stage_a_probe dune_search ${TORCH_LIBRARIES})

  add_executable(dune_debug_replay dune_debug_replay.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_debug_replay dune_search ${TORCH_LIBRARIES})

  # dune_online_search_collector.cc carries the SHARED acceptance rule
  # (ComputeSearchAcceptance / SelectAcceptancePriors, WO-1 Phase 4) that the
  # offline teacher and the online collector both call.
  add_executable(dune_search_teacher
    dune_search_teacher.cc
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_search_teacher dune_search ${TORCH_LIBRARIES})

  add_executable(dune_sm_value_probe dune_sm_value_probe.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_sm_value_probe dune_search ${TORCH_LIBRARIES})

  add_executable(dune_diagnose dune_diagnose.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_diagnose ${TORCH_LIBRARIES})

  add_executable(dune_surgery_12block
    dune_surgery_12block.cc
    dune_surgery_12block_ops.cc
    dune_surgery_12block_verify.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_surgery_12block dune_search ${TORCH_LIBRARIES})

  add_executable(dune_create_corpus dune_create_corpus.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_create_corpus dune_search ${TORCH_LIBRARIES})

  add_executable(dune_checkpoint_policy_identity dune_checkpoint_policy_identity.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_checkpoint_policy_identity ${TORCH_LIBRARIES})

  add_executable(dune_search_calibration dune_search_calibration.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_search_calibration dune_search ${TORCH_LIBRARIES})

  add_executable(dune_repair_probe dune_repair_probe.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_repair_probe dune_search ${TORCH_LIBRARIES})

  add_executable(dune_native_verification_test dune_native_verification_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_native_verification_test ${TORCH_LIBRARIES})

  add_executable(dune_rollout_search_tournament
    dune_rollout_search_tournament.cc
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_rollout_search_tournament dune_search ${TORCH_LIBRARIES})

  add_executable(dune_compound_search_tournament
    dune_compound_search_tournament.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_compound_search_tournament dune_search ${TORCH_LIBRARIES})

  add_executable(dune_play_search_tournament
    dune_play_search_tournament.cc
    dune_play_search_tournament_runner.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_play_search_tournament dune_search ${TORCH_LIBRARIES})

  add_executable(dune_wide_rollout_search_test
    dune_wide_rollout_search_test.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_wide_rollout_search_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_wide_rollout_search_test COMMAND dune_wide_rollout_search_test)

  add_executable(dune_fixed_root_search_benchmark
    dune_fixed_root_search_benchmark.cc
    dune_fixed_root_search_benchmark_setup.cc
    dune_fixed_root_search_benchmark_runner.cc
    dune_fixed_root_search_benchmark_modes.cc
    dune_fixed_root_search_benchmark_replay.cc
    dune_fixed_root_search_benchmark_stats.cc
    dune_fixed_root_search_benchmark_report.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_fixed_root_search_benchmark dune_search ${TORCH_LIBRARIES})

  add_executable(dune_compound_turn_search_test
    dune_compound_turn_search_test.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_compound_turn_search_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_compound_turn_search_test COMMAND dune_compound_turn_search_test)

  add_executable(dune_belief_particles_test
    dune_belief_particles_test.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_belief_particles_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_belief_particles_test COMMAND dune_belief_particles_test)

  add_executable(dune_belief_prediction_check
    dune_belief_prediction_check.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_belief_prediction_check dune_search ${TORCH_LIBRARIES})

  add_executable(dune_compound_search_benchmark
    dune_compound_search_benchmark.cc
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_compound_search_benchmark dune_search ${TORCH_LIBRARIES})

  # PF-2 Part B: the matched-fallback conformance suite. Classification,
  # sampler and ordinal parity against Path B's OBSERVED behaviour, each
  # paired with a discrimination check that the known-wrong alternative
  # actually fails — B1's primary control is only as trustworthy as these.
  add_executable(dune_pf2_matched_fallback_test
    dune_pf2_matched_fallback_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pf2_matched_fallback_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_pf2_matched_fallback_test
    COMMAND dune_pf2_matched_fallback_test)

  # The Memocorders endgame-VP reporting regression. Lives HERE and not in
  # games/dune_imperium/ because that directory's tree hash is a pinned
  # registered quantity (a67f1925...bdd6) asserted by every PWO-5 arm.
  add_executable(dune_endgame_vp_reporting_test
    dune_endgame_vp_reporting_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_endgame_vp_reporting_test ${TORCH_LIBRARIES})
  add_test(NAME dune_endgame_vp_reporting_test
    COMMAND dune_endgame_vp_reporting_test)

  # Invariant tests for the per-event VP attribution ledger. Lives here, not in
  # games/dune_imperium/, for the same tree-hash reason as the test above.
  add_executable(dune_vp_ledger_test
    dune_vp_ledger_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_vp_ledger_test ${TORCH_LIBRARIES})
  add_test(NAME dune_vp_ledger_test COMMAND dune_vp_ledger_test)

  add_executable(dune_seed_utils_test dune_seed_utils_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_seed_utils_test ${TORCH_LIBRARIES})

  add_executable(dune_ppo_training_utils_test dune_ppo_training_utils_test.cc
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_ppo_training_utils_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_ppo_training_utils_test COMMAND dune_ppo_training_utils_test)
  set_tests_properties(dune_ppo_training_utils_test PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/..")

  add_executable(dune_sha256_test dune_sha256_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_sha256_test ${TORCH_LIBRARIES})
  add_test(NAME dune_sha256_test COMMAND dune_sha256_test)

  # PWO-5 gate 2 item (b): the sign of --specimen_exchange_penalty. Links
  # dune_ppo_training_utils.cc so it exercises ApplySpecimenExchangeShaping --
  # the function dune_ppo_train's rollout loop actually calls, not a copy.
  add_executable(dune_shaping_sign_test dune_shaping_sign_test.cc
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_shaping_sign_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_shaping_sign_test COMMAND dune_shaping_sign_test)

  # PWO-5 gate 2 item (d): --opponent_greedy / --opponent_temperature leave the
  # default path bitwise unchanged. Header-only under test, so no torch needed
  # beyond the standard link.
  add_executable(dune_eval_action_selection_test
    dune_eval_action_selection_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_eval_action_selection_test ${TORCH_LIBRARIES})
  add_test(NAME dune_eval_action_selection_test
           COMMAND dune_eval_action_selection_test)

  add_executable(dune_targeted_buy_test dune_targeted_buy_test.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj> ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_targeted_buy_test ${TORCH_LIBRARIES})
  add_test(NAME dune_targeted_buy_test COMMAND dune_targeted_buy_test)


  add_executable(dune_offline_utils_test dune_offline_utils_test.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj> ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_offline_utils_test ${TORCH_LIBRARIES})
  add_test(NAME dune_offline_utils_test COMMAND dune_offline_utils_test)

  add_executable(dune_opponent_pool_test dune_opponent_pool_test.cc
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_opponent_pool_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_opponent_pool_test COMMAND dune_opponent_pool_test)

  add_executable(dune_semantic_verification_test dune_semantic_verification_test.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj>
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_semantic_verification_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_semantic_verification_test COMMAND dune_semantic_verification_test)

  add_executable(dune_bf16_cache_verification_test dune_bf16_cache_verification_test.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj>
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_bf16_cache_verification_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_bf16_cache_verification_test COMMAND dune_bf16_cache_verification_test)

  add_executable(dune_purchase_identity_test
    ../games/dune_imperium/dune_purchase_identity_test.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj> ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_purchase_identity_test ${TORCH_LIBRARIES})
  add_test(NAME dune_purchase_identity_test COMMAND dune_purchase_identity_test)

  add_executable(dune_purchase_identity_network_test
    dune_purchase_identity_network_test.cc
    $<TARGET_OBJECTS:dune_ppo_utils_obj>
    $<TARGET_OBJECTS:dune_collector_obj>
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_purchase_identity_network_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_purchase_identity_network_test COMMAND dune_purchase_identity_network_test)

  add_executable(dune_market_diagnostics_test
    dune_market_diagnostics_test.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj> ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_market_diagnostics_test ${TORCH_LIBRARIES})
  add_test(NAME dune_market_diagnostics_test COMMAND dune_market_diagnostics_test)

  add_executable(dune_frozen_policy_diagnostics_test
    dune_frozen_policy_diagnostics_test.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj> ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_frozen_policy_diagnostics_test ${TORCH_LIBRARIES})
  add_test(NAME dune_frozen_policy_diagnostics_test COMMAND dune_frozen_policy_diagnostics_test)

  add_executable(dune_frozen_policy_diagnostics
    dune_frozen_policy_diagnostics_main.cc
    $<TARGET_OBJECTS:dune_imperium_test_utils_obj> ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_frozen_policy_diagnostics ${TORCH_LIBRARIES})
endif()

add_executable(cfr_example cfr_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(cfr_example_test cfr_example)

if (OPEN_SPIEL_BUILD_WITH_ACPC)
add_executable(universal_poker_mccfr_acpc_gamedef_example universal_poker_mccfr_acpc_gamedef_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(universal_poker_mccfr_acpc_gamedef_example_test universal_poker_mccfr_acpc_gamedef_example)
endif()

add_executable(cfr_multi_equilibria_example cfr_multi_equilibria_example.cc
               ${OPEN_SPIEL_OBJECTS})

add_executable(imperfect_recall_mccfr imperfect_recall_mccfr.cc
               ${OPEN_SPIEL_OBJECTS})

add_executable(example example.cc ${OPEN_SPIEL_OBJECTS})
add_test(example_test example --game=tic_tac_toe --seed=0)

add_executable(fsicfr_liars_dice fsicfr_liars_dice.cc ${OPEN_SPIEL_OBJECTS})

add_executable(gtp gtp.cc ${OPEN_SPIEL_OBJECTS})

add_executable(is_mcts_gwhist is_mcts_gwhist.cc ${OPEN_SPIEL_OBJECTS})

add_executable(matrix_example matrix_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(matrix_example_test matrix_example)

add_executable(mcts_example mcts_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(mcts_example_test mcts_example)

add_executable(minimax_example minimax_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(minimax_example_test minimax_example)

add_executable(policy_iteration_example policy_iteration_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(policy_iteration_example_test policy_iteration_example)

add_executable(value_iteration_example value_iteration_example.cc ${OPEN_SPIEL_OBJECTS})
add_test(value_iteration_example_test value_iteration_example)

add_executable(tabular_sarsa_example tabular_sarsa_example.cc ${OPEN_SPIEL_OBJECTS})

add_executable(tabular_q_learning_example tabular_q_learning_example.cc ${OPEN_SPIEL_OBJECTS})

add_executable(count_all_states count_all_states.cc ${OPEN_SPIEL_OBJECTS})

# if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
#   add_executable(alpha_zero_torch_example alpha_zero_torch_example.cc
#                 ${OPEN_SPIEL_OBJECTS}
#                 $<TARGET_OBJECTS:alpha_zero_torch>)
#   target_link_libraries (alpha_zero_torch_example ${TORCH_LIBRARIES})
#
#   add_executable(alpha_zero_torch_game_example alpha_zero_torch_game_example.cc
#                 ${OPEN_SPIEL_OBJECTS}
#                 $<TARGET_OBJECTS:alpha_zero_torch>
#                 $<TARGET_OBJECTS:bots>)
#   target_link_libraries (alpha_zero_torch_game_example ${TORCH_LIBRARIES})
#
#   add_executable(dqn_torch_example dqn_torch_example.cc
#                 ${OPEN_SPIEL_OBJECTS}
#                 $<TARGET_OBJECTS:dqn_torch>)
#   target_link_libraries (dqn_torch_example ${TORCH_LIBRARIES})
# endif ()

if (BUILD_SHARED_LIB)
  if (WIN32)
    add_executable(shared_library_example shared_library_example.cc ${OPEN_SPIEL_OBJECTS})
  else()
    add_executable(shared_library_example shared_library_example.cc)
  endif()
  target_link_libraries(shared_library_example open_spiel)
  add_test(shared_lib_test shared_lib_test)
endif()
