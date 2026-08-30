#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "dune_sha256.h"
#include "open_spiel/spiel.h"

namespace open_spiel {

inline constexpr int kVrpoNumSeats = 4;
inline constexpr int kVrpoDuneInformationStateSize = 5580;
inline constexpr int kVrpoJointInformationSize =
    kVrpoNumSeats * kVrpoDuneInformationStateSize;
inline constexpr int kVrpoDuneActionDim = 2391;
inline constexpr double kVrpoMaxProbabilityTolerance = 1e-6;
inline constexpr char kVrpoJointInformationEncodingLabel[] =
    "actor_relative_joint_information_proxy_not_full_markov_state_v1";
inline constexpr bool kVrpoJointInformationIsFullMarkovState = false;
inline constexpr bool kVrpoJointInformationMayFeedActorInference = false;

// Privileged training-only proxy. Segment slot s contains the information-state
// tensor for absolute seat (actor+s)%4. It intentionally is NOT called a full
// state: shared hidden deck/order state remains absent from every player's
// information tensor. Phase 1 has no actor or training call site for this
// helper, so it is structurally default-inert.
inline bool ActorRelativeJointInformationTensor(
    const State& state, Player actor, std::vector<float>* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null joint-information output");
  output->clear();
  const std::shared_ptr<const Game> game = state.GetGame();
  if (game == nullptr) return fail("state has no game");
  if (game->GetType().short_name != "dune_imperium") {
    return fail("joint-information encoding requires exact game identity dune_imperium");
  }
  if (game->NumPlayers() != kVrpoNumSeats) {
    return fail("joint-information encoding requires exactly four players");
  }
  if (actor < 0 || actor >= kVrpoNumSeats) {
    return fail("joint-information actor is out of range");
  }
  if (!game->GetType().provides_information_state_tensor) {
    return fail("game does not provide information-state tensors");
  }
  if (game->InformationStateTensorSize() !=
      kVrpoDuneInformationStateSize) {
    return fail("Dune information-state width is not the pinned 5580");
  }
  std::vector<float> joined;
  joined.reserve(kVrpoJointInformationSize);
  for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
    const Player absolute_seat = (actor + slot) % kVrpoNumSeats;
    const std::vector<float> segment =
        state.InformationStateTensor(absolute_seat);
    if (segment.size() !=
        static_cast<size_t>(kVrpoDuneInformationStateSize)) {
      return fail("one information-state segment has the wrong width");
    }
    for (float value : segment) {
      if (!std::isfinite(value)) {
        return fail("joint-information segment contains a nonfinite value");
      }
    }
    joined.insert(joined.end(), segment.begin(), segment.end());
  }
  if (joined.size() != static_cast<size_t>(kVrpoJointInformationSize)) {
    return fail("joint-information output width is not exactly 4*5580");
  }
  *output = std::move(joined);
  return true;
}

inline std::string VrpoJointInformationSha256(
    Player actor, const std::vector<float>& joint_information) {
  std::string payload(kVrpoJointInformationEncodingLabel);
  payload.push_back('\0');
  payload.append(reinterpret_cast<const char*>(&actor), sizeof(actor));
  if (!joint_information.empty()) {
    payload.append(reinterpret_cast<const char*>(joint_information.data()),
                   joint_information.size() * sizeof(float));
  }
  return ComputeStringSHA256(payload);
}

// Every value indexed by seat in the reference arithmetic is in ABSOLUTE seat
// order [seat0, seat1, seat2, seat3]. Actor-relative arrays are a distinct type
// and must pass through the checked adapter below before entering a timeline.
using VrpoSeatValues = std::array<double, kVrpoNumSeats>;

struct VrpoActorRelativeSeatValues {
  // slot 0 = actor; slot s = absolute seat (actor+s)%4.
  std::array<double, kVrpoNumSeats> slots = {0.0, 0.0, 0.0, 0.0};
};

inline bool VrpoActorRelativeToAbsoluteSeatValues(
    Player actor, const VrpoActorRelativeSeatValues& relative,
    VrpoSeatValues* absolute, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (absolute != nullptr) *absolute = {0.0, 0.0, 0.0, 0.0};
    return false;
  };
  if (absolute == nullptr) return fail("null absolute-seat output");
  *absolute = {0.0, 0.0, 0.0, 0.0};
  if (actor < 0 || actor >= kVrpoNumSeats) {
    return fail("relative-seat adapter actor is out of range");
  }
  for (int slot = 0; slot < kVrpoNumSeats; ++slot) {
    const double value = relative.slots[slot];
    if (!std::isfinite(value)) {
      return fail("relative-seat adapter input is nonfinite");
    }
    (*absolute)[(actor + slot) % kVrpoNumSeats] = value;
  }
  return true;
}

// One row in one complete episode's GLOBAL player-decision timeline. Q is
// legal-only and aligned to ordered legal_actions: legal_q_values[a][i] is the
// value to ABSOLUTE seat i of the row actor taking legal action a. Rewards use
// the same absolute [0,1,2,3] order.
struct VrpoTimelineRow {
  uint64_t episode_id = 0;
  Player actor = kInvalidPlayer;
  std::vector<Action> legal_actions;
  int chosen_index = -1;
  Action chosen_action = kInvalidAction;
  std::vector<double> legal_probabilities;
  std::vector<VrpoSeatValues> legal_q_values;
  VrpoSeatValues rewards = {0.0, 0.0, 0.0, 0.0};
  bool terminal_after = false;
};

struct VrpoReferenceRow {
  // All four arrays are in ABSOLUTE seat order [0,1,2,3].
  VrpoSeatValues v = {0.0, 0.0, 0.0, 0.0};
  VrpoSeatValues delta = {0.0, 0.0, 0.0, 0.0};
  VrpoSeatValues g = {0.0, 0.0, 0.0, 0.0};
  VrpoSeatValues q_target = {0.0, 0.0, 0.0, 0.0};
  double actor_advantage = 0.0;
};

struct VrpoReferenceTrace {
  uint64_t episode_id = 0;
  std::vector<VrpoReferenceRow> rows;
  std::string canonical_sha256;
};

inline bool GatherVrpoLegalQValues(
    const std::vector<VrpoSeatValues>& dense_q_values,
    const std::vector<Action>& ordered_legal_actions,
    std::vector<VrpoSeatValues>* legal_q_values, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (legal_q_values != nullptr) legal_q_values->clear();
    return false;
  };
  if (legal_q_values == nullptr) return fail("null legal-Q output");
  legal_q_values->clear();
  if (dense_q_values.size() != static_cast<size_t>(kVrpoDuneActionDim)) {
    return fail("dense Q width is not the Dune action dimension");
  }
  if (ordered_legal_actions.empty()) return fail("legal action list is empty");
  std::set<Action> seen;
  std::vector<VrpoSeatValues> gathered;
  gathered.reserve(ordered_legal_actions.size());
  for (Action action : ordered_legal_actions) {
    if (action < 0 || action >= kVrpoDuneActionDim) {
      return fail("legal action is out of range");
    }
    if (!seen.insert(action).second) {
      return fail("legal action list contains a duplicate");
    }
    const VrpoSeatValues& values = dense_q_values[action];
    if (!std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
      return fail("a legal Q value is nonfinite");
    }
    gathered.push_back(values);
  }
  *legal_q_values = std::move(gathered);
  return true;
}

namespace vrpo_internal {

inline bool ReferenceTraceSha256(
    const std::vector<VrpoTimelineRow>& timeline,
    const std::vector<VrpoReferenceRow>& rows, double gamma,
    double lambda, std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null trace-hash output");
  output->clear();
  if (timeline.size() != rows.size()) {
    return fail("trace-hash input/result row counts differ");
  }
  if (!std::isfinite(gamma) || !std::isfinite(lambda)) {
    return fail("trace-hash gamma or lambda is nonfinite");
  }
  std::string payload = "dune_vrpo_global_expected_sarsa_lambda_v1";
  auto append = [&](const auto& value) {
    payload.append(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  append(gamma);
  append(lambda);
  const uint64_t count = timeline.size();
  append(count);
  for (size_t t = 0; t < timeline.size(); ++t) {
    const VrpoTimelineRow& input = timeline[t];
    const VrpoReferenceRow& reference = rows[t];
    append(input.episode_id);
    append(input.actor);
    append(input.chosen_index);
    append(input.chosen_action);
    append(input.terminal_after);
    const uint64_t legal_count = input.legal_actions.size();
    append(legal_count);
    for (size_t a = 0; a < input.legal_actions.size(); ++a) {
      append(input.legal_actions[a]);
      append(input.legal_probabilities[a]);
      for (double q : input.legal_q_values[a]) append(q);
    }
    for (double reward : input.rewards) append(reward);
    for (double value : reference.v) append(value);
    for (double value : reference.delta) append(value);
    for (double value : reference.g) append(value);
    for (double value : reference.q_target) append(value);
    append(reference.actor_advantage);
    const auto finite = [](const VrpoSeatValues& values) {
      return std::all_of(values.begin(), values.end(),
                         [](double value) { return std::isfinite(value); });
    };
    if (!finite(reference.v) || !finite(reference.delta) ||
        !finite(reference.g) || !finite(reference.q_target) ||
        !std::isfinite(reference.actor_advantage)) {
      return fail("trace-hash reference row is nonfinite");
    }
  }
  *output = ComputeStringSHA256(payload);
  return true;
}

}  // namespace vrpo_internal

// Literal paper-equation reference over the GLOBAL decision timeline:
//   V[t,i]     = sum_a pi[t,a] Q[t,a,i]
//   delta[t,i] = r[t,i] + gamma V[t+1,i] - Qchosen[t,i]
//   G[t,i]     = delta[t,i] + gamma lambda G[t+1,i]
//   A_actor[t] = Qchosen[t,actor] - V[t,actor] + G[t,actor]
//   Qtarget    = Qchosen + G
// The t+1 row is used even when an opponent acts; only the final row has zero
// next V. Taking exactly one episode prevents cross-episode leakage.
inline bool ComputeVrpoExpectedSarsaLambdaReference(
    const std::vector<VrpoTimelineRow>& timeline, double gamma,
    double lambda, VrpoReferenceTrace* output, std::string* error,
    double probability_tolerance = 1e-9) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoReferenceTrace{};
    return false;
  };
  if (output == nullptr) return fail("null VRPO reference output");
  *output = VrpoReferenceTrace{};
  if (timeline.empty()) return fail("episode timeline is empty");
  if (!std::isfinite(gamma) || gamma < 0.0 || gamma > 1.0 ||
      !std::isfinite(lambda) || lambda < 0.0 || lambda > 1.0) {
    return fail("gamma and lambda must be finite in [0,1]");
  }
  if (!std::isfinite(probability_tolerance) ||
      probability_tolerance < 0.0 ||
      probability_tolerance > kVrpoMaxProbabilityTolerance) {
    return fail("probability tolerance is invalid");
  }
  const uint64_t episode_id = timeline.front().episode_id;
  std::vector<VrpoReferenceRow> rows(timeline.size());
  std::vector<VrpoSeatValues> chosen_q(timeline.size());
  for (size_t t = 0; t < timeline.size(); ++t) {
    const VrpoTimelineRow& row = timeline[t];
    if (row.episode_id != episode_id) {
      return fail("timeline crosses an episode boundary");
    }
    if (row.actor < 0 || row.actor >= kVrpoNumSeats) {
      return fail("row actor is out of range");
    }
    if (row.terminal_after != (t + 1 == timeline.size())) {
      return fail("terminal boundary must occur exactly after the final row");
    }
    const size_t legal_count = row.legal_actions.size();
    if (legal_count == 0) return fail("row legal action list is empty");
    if (row.legal_probabilities.size() != legal_count ||
        row.legal_q_values.size() != legal_count) {
      return fail("legal action/probability/Q widths differ");
    }
    if (row.chosen_index < 0 ||
        row.chosen_index >= static_cast<int>(legal_count) ||
        row.legal_actions[row.chosen_index] != row.chosen_action) {
      return fail("chosen index/action is not exactly aligned to legal order");
    }
    std::set<Action> seen;
    double probability_sum = 0.0;
    for (size_t a = 0; a < legal_count; ++a) {
      const Action action = row.legal_actions[a];
      if (action < 0 || action >= kVrpoDuneActionDim) {
        return fail("row contains an out-of-range legal action");
      }
      if (!seen.insert(action).second) {
        return fail("row contains a duplicate legal action");
      }
      const double probability = row.legal_probabilities[a];
      if (!std::isfinite(probability) || probability < 0.0 ||
          probability > 1.0 + probability_tolerance) {
        return fail("row contains an invalid legal probability");
      }
      probability_sum += probability;
      for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
        const double q = row.legal_q_values[a][seat];
        if (!std::isfinite(q)) return fail("row contains a nonfinite legal Q");
        rows[t].v[seat] += probability * q;
      }
    }
    if (!std::isfinite(probability_sum) ||
        std::abs(probability_sum - 1.0) > probability_tolerance) {
      return fail("legal probabilities do not sum to one");
    }
    if (!std::all_of(rows[t].v.begin(), rows[t].v.end(),
                     [](double value) { return std::isfinite(value); })) {
      return fail("derived V is nonfinite");
    }
    for (double reward : row.rewards) {
      if (!std::isfinite(reward)) return fail("row contains a nonfinite reward");
    }
    chosen_q[t] = row.legal_q_values[row.chosen_index];
  }

  for (size_t t = 0; t < timeline.size(); ++t) {
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      const double next_v =
          (t + 1 < timeline.size()) ? rows[t + 1].v[seat] : 0.0;
      rows[t].delta[seat] = timeline[t].rewards[seat] + gamma * next_v -
                            chosen_q[t][seat];
      if (!std::isfinite(rows[t].delta[seat])) {
        return fail("derived delta is nonfinite");
      }
    }
  }
  VrpoSeatValues next_g = {0.0, 0.0, 0.0, 0.0};
  for (size_t reverse = timeline.size(); reverse-- > 0;) {
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      rows[reverse].g[seat] = rows[reverse].delta[seat] +
                              gamma * lambda * next_g[seat];
      if (!std::isfinite(rows[reverse].g[seat])) {
        return fail("derived reverse G is nonfinite");
      }
      rows[reverse].q_target[seat] =
          chosen_q[reverse][seat] + rows[reverse].g[seat];
      if (!std::isfinite(rows[reverse].q_target[seat])) {
        return fail("derived Q target is nonfinite");
      }
    }
    const int actor = timeline[reverse].actor;
    rows[reverse].actor_advantage =
        chosen_q[reverse][actor] - rows[reverse].v[actor] +
        rows[reverse].g[actor];
    if (!std::isfinite(rows[reverse].actor_advantage)) {
      return fail("derived actor advantage is nonfinite");
    }
    next_g = rows[reverse].g;
  }

  VrpoReferenceTrace result;
  result.episode_id = episode_id;
  result.rows = std::move(rows);
  std::string hash_error;
  if (!vrpo_internal::ReferenceTraceSha256(
          timeline, result.rows, gamma, lambda, &result.canonical_sha256,
          &hash_error)) {
    return fail("trace hash failed: " + hash_error);
  }
  *output = std::move(result);
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_H_
