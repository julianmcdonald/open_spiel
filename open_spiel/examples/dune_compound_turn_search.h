// Copyright 2026 DeepMind Technologies Limited.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <unordered_map>
#include <vector>

#include "open_spiel/examples/dune_batched_evaluator.h"
#include "open_spiel/examples/dune_online_search_collector.h"
#include "open_spiel/examples/dune_seed_utils.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/games/dune_imperium/dune_imperium_cards.h"
#include "open_spiel/games/dune_imperium/dune_imperium_common.h"
#include "open_spiel/games/dune_imperium/dune_imperium_util.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "open_spiel/examples/dune_compound_turn_tree.h"
#include "open_spiel/examples/dune_compound_turn_stats.h"
#include "open_spiel/examples/dune_compound_turn_rollout.h"
#include "open_spiel/examples/dune_compound_turn_decision_helpers.h"
#include "open_spiel/examples/dune_compound_turn_decision.h"
#include "open_spiel/examples/dune_compound_turn_batch.h"

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_COMPOUND_TURN_SEARCH_H_
