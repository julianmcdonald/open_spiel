add_executable(benchmark_game benchmark_game.cc ${OPEN_SPIEL_OBJECTS})
add_test(benchmark_game_test benchmark_game --game=tic_tac_toe --sims=100 --attempts=2)

if (OPEN_SPIEL_BUILD_WITH_LIBTORCH)
  add_executable(dune_benchmark dune_benchmark.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_benchmark ${TORCH_LIBRARIES})

  add_executable(dune_ppo_train dune_ppo_train.cc dune_ppo_training_utils.cc ${OPEN_SPIEL_OBJECTS})
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

  add_executable(dune_eval_1000 dune_eval_1000.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_eval_1000 ${TORCH_LIBRARIES})

  add_executable(dune_population_eval dune_population_eval.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_population_eval ${TORCH_LIBRARIES})

  add_executable(dune_search_benchmark dune_search_benchmark.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_search_benchmark dune_search ${TORCH_LIBRARIES})

  add_executable(dune_search_teacher dune_search_teacher.cc ${OPEN_SPIEL_OBJECTS})
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

  add_executable(dune_seed_utils_test dune_seed_utils_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_seed_utils_test ${TORCH_LIBRARIES})

  add_executable(dune_ppo_training_utils_test dune_ppo_training_utils_test.cc dune_ppo_training_utils.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_ppo_training_utils_test ${TORCH_LIBRARIES})

  add_executable(dune_sha256_test dune_sha256_test.cc ${OPEN_SPIEL_OBJECTS})
  target_link_libraries(dune_sha256_test ${TORCH_LIBRARIES})
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
