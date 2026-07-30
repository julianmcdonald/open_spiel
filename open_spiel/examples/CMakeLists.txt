add_executable(benchmark_game benchmark_game.cc ${OPEN_SPIEL_OBJECTS})
add_test(benchmark_game_test benchmark_game --game=tic_tac_toe --sims=100 --attempts=2)

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_executable(dune_benchmark dune_benchmark.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_benchmark ${TORCH_LIBRARIES})

  add_executable(dune_ppo_train dune_ppo_train.cc dune_ppo_training_utils.cc dune_online_search_collector.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_ppo_train dune_search ${TORCH_LIBRARIES})

  add_executable(dune_evaluator dune_evaluator.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_evaluator ${TORCH_LIBRARIES})

  add_executable(dune_evaluator_test dune_evaluator_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_evaluator_test ${TORCH_LIBRARIES})

  add_library(dune_search STATIC dune_puct_is_mcts.cc dune_search_routing.cc dune_search_session.cc)
  target_link_libraries(dune_search PRIVATE ${TORCH_LIBRARIES})
  target_include_directories(dune_search PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(dune_puct_is_mcts_test dune_puct_is_mcts_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_puct_is_mcts_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_puct_is_mcts_test COMMAND dune_puct_is_mcts_test)

  # WO-1 Phase 4: teacher/collector acceptance parity on a shared fixture.
  add_executable(dune_acceptance_parity_test
    dune_acceptance_parity_test.cc dune_online_search_collector.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_acceptance_parity_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_acceptance_parity_test COMMAND dune_acceptance_parity_test)

  add_executable(dune_online_search_collector_test
    dune_online_search_collector_test.cc dune_online_search_collector.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_online_search_collector_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_online_search_collector_test COMMAND dune_online_search_collector_test)

  add_executable(dune_collector_smoke
    dune_collector_smoke.cc dune_online_search_collector.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_collector_smoke dune_search ${TORCH_LIBRARIES})

  add_executable(dune_eval_1000 dune_eval_1000.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_eval_1000 ${TORCH_LIBRARIES})

  add_executable(dune_population_eval dune_population_eval.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_population_eval ${TORCH_LIBRARIES})

  add_executable(dune_search_benchmark dune_search_benchmark.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_search_benchmark dune_search ${TORCH_LIBRARIES})

  # dune_online_search_collector.cc carries the SHARED acceptance rule
  # (ComputeSearchAcceptance / SelectAcceptancePriors, WO-1 Phase 4) that the
  # offline teacher and the online collector both call.
  add_executable(dune_search_teacher dune_search_teacher.cc dune_online_search_collector.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_search_teacher dune_search ${TORCH_LIBRARIES})

  add_executable(dune_sm_value_probe dune_sm_value_probe.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_sm_value_probe dune_search ${TORCH_LIBRARIES})

  add_executable(dune_diagnose dune_diagnose.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_diagnose ${TORCH_LIBRARIES})

  add_executable(dune_create_corpus dune_create_corpus.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_create_corpus dune_search ${TORCH_LIBRARIES})

  add_executable(dune_critic_fidelity_gate dune_critic_fidelity_gate.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_critic_fidelity_gate dune_search ${TORCH_LIBRARIES})

  add_executable(dune_checkpoint_policy_identity dune_checkpoint_policy_identity.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_checkpoint_policy_identity ${TORCH_LIBRARIES})

  add_executable(dune_search_calibration dune_search_calibration.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_search_calibration dune_search ${TORCH_LIBRARIES})

  add_executable(dune_pwo2_root_corpus dune_pwo2_root_corpus.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo2_root_corpus dune_search ${TORCH_LIBRARIES})

  add_executable(dune_pwo2_qualify dune_pwo2_qualify.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo2_qualify dune_search ${TORCH_LIBRARIES})

  add_executable(dune_pwo3_strata dune_pwo3_strata.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo3_strata dune_search ${TORCH_LIBRARIES})

  add_executable(dune_pwo3_teacher_audit dune_pwo3_teacher_audit.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo3_teacher_audit dune_search ${TORCH_LIBRARIES})

  # PWO-3 Amendment 2 (docs/PWO3_AMENDMENT_2_PRECISION_RECOVERY.md). A SEPARATE
  # target on purpose: dune_pwo3_teacher_audit is the binary of record for every
  # PWO-3 measurement and must survive the amendment byte-for-byte, so the
  # recovery instrument may not be a mode on it.
  add_executable(dune_pwo3_prior_recovery dune_pwo3_prior_recovery.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo3_prior_recovery dune_search ${TORCH_LIBRARIES})

  # PWO-4 trajectory distillation (docs/PWO4_TRAJECTORY_REGISTRATION.md rev 3,
  # frozen by docs/PWO4_FREEZE_ADDENDUM.md). A SEPARATE target for the same reason
  # dune_pwo3_prior_recovery is separate from dune_pwo3_teacher_audit: here the
  # binary of record is dune_search_benchmark, the reference instrument of the
  # section 7.1 cross-build fidelity gate (sha256 60ea430a...). It must survive
  # byte-for-byte, so the generator may not be a mode on it, and `make` must never
  # relink it. Build ONLY this target.
  add_executable(dune_pwo4_trajectory dune_pwo4_trajectory.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo4_trajectory dune_search ${TORCH_LIBRARIES})

  # CP0.6 guard 3: decision_role must equal an INDEPENDENTLY reconstructed
  # classification. A second call on the emitter's own live state would prove only
  # that the classifier is deterministic, so the check is a separate binary that
  # replays each decision's persisted history from a fresh initial state and
  # reclassifies. Loads no model and runs no search.
  add_executable(dune_pwo4_role_verify dune_pwo4_role_verify.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo4_role_verify dune_search ${TORCH_LIBRARIES})

  # Battery check 9: root raw priors are immutable after expansion, which is what
  # makes the recovery instrument's zero-simulation replay admissible.
  # PWO-5 gate 3: the offline preparation pass. Replays all 400 PWO-4 game
  # histories through the frozen engine, runs the section 5.3 replay-fidelity
  # gate over all 48,443 label rows, derives the three auxiliary targets, fixes
  # the section 8.5 whole-trajectory split, and writes the distillation label
  # pack plus the parallel hashed auxiliary-target artifact.
  add_executable(dune_pwo5_prepare dune_pwo5_prepare.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo5_prepare dune_search ${TORCH_LIBRARIES})

  # PWO-5 amendment 1 ruling 7: the update-300 whole-dataset head evaluator.
  # A SEPARATE BINARY on purpose -- Appendix A.1's closing rule makes any
  # gate-3 trainer flag outside its table a STOP, and a whole-dataset
  # evaluation pass has no registered trainer flag.
  add_executable(dune_pwo5_head_eval dune_pwo5_head_eval.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo5_head_eval dune_search ${TORCH_LIBRARIES})

  # PWO-5 gate 3: the auxiliary-label loader's manifest FILE ROLES
  # (amendment 1 ruling 5) and the numeric head losses.
  add_executable(dune_pwo5_loader_roles_test dune_pwo5_loader_roles_test.cc
    dune_ppo_training_utils.cc dune_online_search_collector.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo5_loader_roles_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_pwo5_loader_roles_test COMMAND dune_pwo5_loader_roles_test)

  # PWO-5 gate 3: the section 8.6 auxiliary sampler's EXACT pinned algorithm.
  add_executable(dune_pwo5_sampler_test dune_pwo5_sampler_test.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo5_sampler_test ${TORCH_LIBRARIES})
  add_test(NAME dune_pwo5_sampler_test COMMAND dune_pwo5_sampler_test)

  # PWO-5 gate 3: diagnostics schema v5 and the section 14.1c partition identity.
  add_executable(dune_pwo5_schema_test dune_pwo5_schema_test.cc
    dune_ppo_training_utils.cc dune_online_search_collector.cc
    ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo5_schema_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_pwo5_schema_test COMMAND dune_pwo5_schema_test)

  # The Memocorders endgame-VP reporting regression. Lives HERE and not in
  # games/dune_imperium/ because that directory's tree hash is a pinned
  # registered quantity (a67f1925...bdd6) asserted by every PWO-5 arm.
  add_executable(dune_endgame_vp_reporting_test
    dune_endgame_vp_reporting_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_endgame_vp_reporting_test ${TORCH_LIBRARIES})
  add_test(NAME dune_endgame_vp_reporting_test
    COMMAND dune_endgame_vp_reporting_test)

  add_executable(dune_pwo3_immutability_test dune_pwo3_immutability_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_pwo3_immutability_test dune_search ${TORCH_LIBRARIES})
  add_test(NAME dune_pwo3_immutability_test COMMAND dune_pwo3_immutability_test)

  add_executable(dune_seed_utils_test dune_seed_utils_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_seed_utils_test ${TORCH_LIBRARIES})

  add_executable(dune_fidelity_seeds_test dune_fidelity_seeds_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_fidelity_seeds_test ${TORCH_LIBRARIES})
  add_test(NAME dune_fidelity_seeds_test COMMAND dune_fidelity_seeds_test)

  add_executable(dune_ppo_training_utils_test dune_ppo_training_utils_test.cc dune_ppo_training_utils.cc dune_online_search_collector.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_ppo_training_utils_test dune_search ${TORCH_LIBRARIES})

  add_executable(dune_sha256_test dune_sha256_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_sha256_test ${TORCH_LIBRARIES})

  # PWO-5 gate 2 item (b): the sign of --specimen_exchange_penalty. Links
  # dune_ppo_training_utils.cc so it exercises ApplySpecimenExchangeShaping --
  # the function dune_ppo_train's rollout loop actually calls, not a copy.
  add_executable(dune_shaping_sign_test dune_shaping_sign_test.cc
    dune_ppo_training_utils.cc dune_online_search_collector.cc
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

  add_executable(dune_offline_utils_test dune_offline_utils_test.cc
    ../games/dune_imperium/dune_imperium_test_utils.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_offline_utils_test ${TORCH_LIBRARIES})
  add_test(NAME dune_offline_utils_test COMMAND dune_offline_utils_test)
else()
  add_executable(dune_benchmark dune_benchmark.cc ${OPEN_SPIEL_OBJECTS})
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
