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
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH
#include <torch/torch.h>

#include "dune_seed_utils.h"
#endif

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
static_assert(kVrpoDuneInformationStateSize ==
              dune_imperium::kVrpoCentralActorPrefixSize);

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

inline bool VrpoCentralCriticTensorSha256(
    Player actor, const std::vector<float>& central_tensor,
    std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null central-tensor hash output");
  output->clear();
  if (actor < 0 || actor >= kVrpoNumSeats) {
    return fail("central-tensor hash actor is out of range");
  }
  if (central_tensor.size() !=
      static_cast<size_t>(dune_imperium::kVrpoCentralCriticTensorSize)) {
    return fail("central-tensor hash input width is not 9012");
  }
  if (!std::all_of(central_tensor.begin(), central_tensor.end(),
                   [](float value) { return std::isfinite(value); })) {
    return fail("central-tensor hash input is nonfinite");
  }
  std::string payload(dune_imperium::kVrpoCentralCriticTensorSchemaSha256);
  payload.push_back('\0');
  payload.append(reinterpret_cast<const char*>(&actor), sizeof(actor));
  payload.append(reinterpret_cast<const char*>(central_tensor.data()),
                 central_tensor.size() * sizeof(float));
  *output = ComputeStringSHA256(payload);
  return true;
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

// Checked Q boundary: the canonical Q module emits relative slots because its
// input is actor-relative; paper-reference rows accept absolute seat order
// only. Phase-2 callers must convert every legal action through this adapter.
inline bool VrpoActorRelativeQToAbsolute(
    Player actor,
    const std::vector<VrpoActorRelativeSeatValues>& relative_legal_q,
    std::vector<VrpoSeatValues>* absolute_legal_q, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (absolute_legal_q != nullptr) absolute_legal_q->clear();
    return false;
  };
  if (absolute_legal_q == nullptr) return fail("null absolute legal-Q output");
  absolute_legal_q->clear();
  if (relative_legal_q.empty()) return fail("relative legal-Q input is empty");
  std::vector<VrpoSeatValues> converted;
  converted.reserve(relative_legal_q.size());
  for (const auto& relative : relative_legal_q) {
    VrpoSeatValues absolute;
    std::string adapter_error;
    if (!VrpoActorRelativeToAbsoluteSeatValues(
            actor, relative, &absolute, &adapter_error)) {
      return fail("relative legal-Q conversion failed: " + adapter_error);
    }
    converted.push_back(absolute);
  }
  *absolute_legal_q = std::move(converted);
  return true;
}

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

inline constexpr int kVrpoQHiddenDim = 512;
inline constexpr int kVrpoQNumResBlocks = 2;
inline constexpr double kVrpoQFinalInitScale = 0.01;

struct DuneVrpoQResBlockImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::LayerNorm ln1{nullptr};
  torch::nn::LayerNorm ln2{nullptr};

  DuneVrpoQResBlockImpl() {
    fc1 = register_module(
        "fc1", torch::nn::Linear(kVrpoQHiddenDim, kVrpoQHiddenDim));
    fc2 = register_module(
        "fc2", torch::nn::Linear(kVrpoQHiddenDim, kVrpoQHiddenDim));
    ln1 = register_module(
        "ln1", torch::nn::LayerNorm(
                   torch::nn::LayerNormOptions({kVrpoQHiddenDim})));
    ln2 = register_module(
        "ln2", torch::nn::LayerNorm(
                   torch::nn::LayerNormOptions({kVrpoQHiddenDim})));
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(ln1->forward(fc1->forward(x)));
    x = ln2->forward(fc2->forward(x));
    return torch::relu(x + residual);
  }
};
TORCH_MODULE(DuneVrpoQResBlock);

// Canonical matched Q module for every future 2x2 arm. The final dimension is
// ACTOR-RELATIVE slot order; callers must use VrpoActorRelativeQToAbsolute
// before constructing reference rows. No tanh, dropout, or batch norm.
struct DuneVrpoQNetImpl : torch::nn::Module {
  torch::nn::Linear input_layer{nullptr};
  DuneVrpoQResBlock res1{nullptr};
  DuneVrpoQResBlock res2{nullptr};
  torch::nn::Linear q_head{nullptr};
  uint64_t init_seed = 0;

  explicit DuneVrpoQNetImpl(uint64_t registered_init_seed)
      : init_seed(registered_init_seed) {
    input_layer = register_module(
        "input_layer",
        torch::nn::Linear(dune_imperium::kVrpoCentralCriticTensorSize,
                          kVrpoQHiddenDim));
    res1 = register_module("res1", DuneVrpoQResBlock());
    res2 = register_module("res2", DuneVrpoQResBlock());
    q_head = register_module(
        "q_head",
        torch::nn::Linear(kVrpoQHiddenDim,
                          kVrpoDuneActionDim * kVrpoNumSeats));

    torch::NoGradGuard no_grad;
    at::Generator generator =
        dune_seed::MakeTorchCPUGenerator(registered_init_seed);
    auto initialize_linear = [&](torch::nn::Linear& layer, double scale) {
      const double bound = 1.0 / std::sqrt(layer->weight.size(1));
      layer->weight.uniform_(-bound, bound, generator);
      layer->weight.mul_(scale);
      if (layer->bias.defined()) layer->bias.zero_();
    };
    initialize_linear(input_layer, 1.0);
    for (DuneVrpoQResBlock block : {res1, res2}) {
      initialize_linear(block->fc1, 1.0);
      initialize_linear(block->fc2, 1.0);
      block->ln1->weight.fill_(1.0);
      block->ln1->bias.zero_();
      block->ln2->weight.fill_(1.0);
      block->ln2->bias.zero_();
    }
    initialize_linear(q_head, kVrpoQFinalInitScale);
  }

  bool ForwardChecked(const torch::Tensor& input, torch::Tensor* output,
                      std::string* error) {
    auto fail = [&](const std::string& message) {
      if (error != nullptr) *error = message;
      if (output != nullptr) *output = torch::Tensor();
      return false;
    };
    if (output == nullptr) return fail("null Q output");
    *output = torch::Tensor();
    if (!input.defined() || input.dim() != 2 || input.size(0) <= 0 ||
        input.size(1) != dune_imperium::kVrpoCentralCriticTensorSize) {
      return fail("Q input must have shape [positive_batch,9012]");
    }
    if (input.scalar_type() != torch::kFloat32) {
      return fail("Q input dtype must be Float32");
    }
    if (input.device() != input_layer->weight.device()) {
      return fail("Q input and module devices differ");
    }
    if (!torch::isfinite(input).all().item<bool>()) {
      return fail("Q input contains a nonfinite value");
    }
    torch::Tensor hidden = torch::relu(input_layer->forward(input));
    hidden = res1->forward(hidden);
    hidden = res2->forward(hidden);
    torch::Tensor values = q_head->forward(hidden).reshape(
        {input.size(0), kVrpoDuneActionDim, kVrpoNumSeats});
    if (values.dim() != 3 || values.size(0) != input.size(0) ||
        values.size(1) != kVrpoDuneActionDim ||
        values.size(2) != kVrpoNumSeats) {
      return fail("Q output shape is not [batch,2391,4]");
    }
    if (!torch::isfinite(values).all().item<bool>()) {
      return fail("Q output contains a nonfinite value");
    }
    *output = values;
    return true;
  }
};
TORCH_MODULE(DuneVrpoQNet);

inline bool VrpoQModuleParameterSha256(
    DuneVrpoQNetImpl& model, std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null Q parameter-hash output");
  output->clear();
  std::string payload = "dune_vrpo_q_module_v1";
  for (const auto& item : model.named_parameters()) {
    torch::Tensor tensor =
        item.value().detach().contiguous().cpu().to(torch::kFloat32);
    if (!torch::isfinite(tensor).all().item<bool>()) {
      return fail("Q module parameter is nonfinite: " + item.key());
    }
    payload.append(item.key());
    payload.push_back('\0');
    for (int64_t dim : tensor.sizes()) {
      payload.append(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }
    payload.push_back('\0');
    payload.append(reinterpret_cast<const char*>(tensor.data_ptr<float>()),
                   tensor.numel() * sizeof(float));
  }
  *output = ComputeStringSHA256(payload);
  return true;
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

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

inline constexpr char kVrpoCaptureSchemaLabel[] =
    "dune_vrpo_capture_v1|episode_id:uint64|global_row_index:int64_contiguous_zero_based|actor:absolute_0_3|actor_obs:float32x5580|central:float32x9012|central_schema_sha256|legal_ids:ordered_unique_in_range|chosen_index_action|legal_behavior_probabilities:float64|rewards:absolute_float64x4|terminal_after|counterfactual=false|row_sha256";
inline constexpr char kVrpoCaptureSchemaSha256[] =
    "372e366961abf192da908041488a4f081cd01e7f34b301867c54f0cc838c6d19";
inline constexpr char kVrpoZeroShapingRewardConventionLabel[] =
    "dune_vrpo_reward_v1|zero_shaping_only|reward_scale=4|intermediate_rewards=absolute_zero_x4|final_reward=clamp(terminal_return_per_absolute_seat/4,-1,1)|gamma_lambda_registered|per_seat_conservation";
inline constexpr char kVrpoZeroShapingRewardConventionSha256[] =
    "b63478ccd2c7bc85028b6659a75c290df8f284a4a45a0fb2bd92abc477d89a9f";

struct VrpoCapturedRow {
  uint64_t episode_id = 0;
  int64_t global_row_index = -1;
  Player actor = kInvalidPlayer;  // absolute seat
  std::vector<float> actor_observation;  // exactly 5580
  std::vector<float> central_tensor;     // exactly 9012
  std::string central_schema_sha256;
  std::vector<Action> legal_actions;
  int chosen_index = -1;
  Action chosen_action = kInvalidAction;
  std::vector<double> legal_behavior_probabilities;
  VrpoSeatValues rewards = {0.0, 0.0, 0.0, 0.0};  // absolute seats
  bool terminal_after = false;
  bool is_counterfactual = false;
  std::string row_sha256;
};

struct VrpoCapturedEpisode {
  uint64_t episode_id = 0;
  std::vector<VrpoCapturedRow> rows;
  std::string capture_schema_sha256;
  std::string reward_convention_sha256;
  double reward_scale = 4.0;
  double gamma = 1.0;
  double lambda = 1.0;
  double probability_tolerance = 1e-9;
  VrpoSeatValues terminal_returns = {0.0, 0.0, 0.0, 0.0};
  bool rewards_finalized = false;
  std::string capture_sha256;
};

struct VrpoZeroShapingRewardConfig {
  double reward_scale = 4.0;
  double gamma = 1.0;
  double lambda = 1.0;
  double probability_tolerance = 1e-9;
  double shaped_reward_weight = 0.0;
  double tleilaxu_breadcrumb_weight = 0.0;
  double tleilaxu_level7_breadcrumb_weight = 0.0;
  double specimen_exchange_penalty = 0.0;
};

namespace vrpo_capture_internal {

template <typename T>
inline void AppendPod(std::string* payload, const T& value) {
  payload->append(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline bool ValidateRowCore(const VrpoCapturedEpisode& episode,
                            const VrpoCapturedRow& row, size_t expected_index,
                            std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (row.episode_id != episode.episode_id) {
    return fail("captured row crosses an episode boundary");
  }
  if (row.global_row_index != static_cast<int64_t>(expected_index)) {
    return fail("captured global row order is not contiguous zero-based");
  }
  if (row.actor < 0 || row.actor >= kVrpoNumSeats) {
    return fail("captured row actor is out of range");
  }
  if (row.is_counterfactual || row.chosen_action == kInvalidAction) {
    return fail("captured row is invalid or counterfactual");
  }
  if (row.actor_observation.size() !=
          static_cast<size_t>(kVrpoDuneInformationStateSize) ||
      row.central_tensor.size() != static_cast<size_t>(
          dune_imperium::kVrpoCentralCriticTensorSize)) {
    return fail("captured actor/central tensor dimensions are invalid");
  }
  if (row.central_schema_sha256 !=
      dune_imperium::kVrpoCentralCriticTensorSchemaSha256) {
    return fail("captured central tensor schema hash is wrong");
  }
  const auto finite_float = [](float value) { return std::isfinite(value); };
  if (!std::all_of(row.actor_observation.begin(),
                   row.actor_observation.end(), finite_float) ||
      !std::all_of(row.central_tensor.begin(), row.central_tensor.end(),
                   finite_float)) {
    return fail("captured actor/central tensor is nonfinite");
  }
  if (std::memcmp(row.actor_observation.data(), row.central_tensor.data(),
                  kVrpoDuneInformationStateSize * sizeof(float)) != 0) {
    return fail("captured central prefix does not bit-match actor observation");
  }
  const size_t legal_count = row.legal_actions.size();
  if (legal_count == 0 ||
      row.legal_behavior_probabilities.size() != legal_count) {
    return fail("captured legal/probability widths differ or are empty");
  }
  if (row.chosen_index < 0 ||
      row.chosen_index >= static_cast<int>(legal_count) ||
      row.legal_actions[row.chosen_index] != row.chosen_action) {
    return fail("captured chosen index/action is not aligned to legal order");
  }
  std::set<Action> seen;
  double probability_sum = 0.0;
  for (size_t a = 0; a < legal_count; ++a) {
    const Action action = row.legal_actions[a];
    if (action < 0 || action >= kVrpoDuneActionDim) {
      return fail("captured legal action is out of range");
    }
    if (!seen.insert(action).second) {
      return fail("captured legal action is duplicated");
    }
    const double probability = row.legal_behavior_probabilities[a];
    if (!std::isfinite(probability) || probability < 0.0 ||
        probability > 1.0 + episode.probability_tolerance) {
      return fail("captured legal probability is invalid");
    }
    probability_sum += probability;
  }
  if (!std::isfinite(probability_sum) ||
      std::abs(probability_sum - 1.0) > episode.probability_tolerance) {
    return fail("captured legal probabilities do not sum to one");
  }
  if (!std::all_of(row.rewards.begin(), row.rewards.end(),
                   [](double value) { return std::isfinite(value); })) {
    return fail("captured absolute reward is nonfinite");
  }
  return true;
}

inline std::string RowSha256(const VrpoCapturedRow& row) {
  std::string payload = kVrpoCaptureSchemaSha256;
  AppendPod(&payload, row.episode_id);
  AppendPod(&payload, row.global_row_index);
  AppendPod(&payload, row.actor);
  payload.append(reinterpret_cast<const char*>(row.actor_observation.data()),
                 row.actor_observation.size() * sizeof(float));
  payload.append(reinterpret_cast<const char*>(row.central_tensor.data()),
                 row.central_tensor.size() * sizeof(float));
  payload.append(row.central_schema_sha256);
  const uint64_t legal_count = row.legal_actions.size();
  AppendPod(&payload, legal_count);
  for (size_t a = 0; a < row.legal_actions.size(); ++a) {
    AppendPod(&payload, row.legal_actions[a]);
    AppendPod(&payload, row.legal_behavior_probabilities[a]);
  }
  AppendPod(&payload, row.chosen_index);
  AppendPod(&payload, row.chosen_action);
  for (double reward : row.rewards) AppendPod(&payload, reward);
  AppendPod(&payload, row.terminal_after);
  AppendPod(&payload, row.is_counterfactual);
  return ComputeStringSHA256(payload);
}

inline std::string EpisodeSha256(const VrpoCapturedEpisode& episode) {
  std::string payload = kVrpoCaptureSchemaSha256;
  payload.append(episode.reward_convention_sha256);
  AppendPod(&payload, episode.episode_id);
  AppendPod(&payload, episode.reward_scale);
  AppendPod(&payload, episode.gamma);
  AppendPod(&payload, episode.lambda);
  AppendPod(&payload, episode.probability_tolerance);
  for (double value : episode.terminal_returns) AppendPod(&payload, value);
  AppendPod(&payload, episode.rewards_finalized);
  const uint64_t rows = episode.rows.size();
  AppendPod(&payload, rows);
  for (const auto& row : episode.rows) payload.append(row.row_sha256);
  return ComputeStringSHA256(payload);
}

inline void RefreshHashes(VrpoCapturedEpisode* episode) {
  for (auto& row : episode->rows) row.row_sha256 = RowSha256(row);
  episode->capture_sha256 = EpisodeSha256(*episode);
}

}  // namespace vrpo_capture_internal

inline bool ValidateVrpoCapturedEpisode(const VrpoCapturedEpisode& episode,
                                        std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (episode.capture_schema_sha256 != kVrpoCaptureSchemaSha256) {
    return fail("captured episode schema hash is wrong");
  }
  if (episode.reward_convention_sha256 !=
      kVrpoZeroShapingRewardConventionSha256) {
    return fail("captured reward convention hash is wrong");
  }
  if (!episode.rewards_finalized) {
    return fail("captured episode rewards are not finalized");
  }
  if (episode.rows.empty()) return fail("captured episode has no rows");
  if (episode.reward_scale != 4.0 || !std::isfinite(episode.gamma) ||
      episode.gamma < 0.0 || episode.gamma > 1.0 ||
      !std::isfinite(episode.lambda) || episode.lambda < 0.0 ||
      episode.lambda > 1.0 ||
      !std::isfinite(episode.probability_tolerance) ||
      episode.probability_tolerance < 0.0 ||
      episode.probability_tolerance > kVrpoMaxProbabilityTolerance) {
    return fail("captured reward scale/gamma/lambda/tolerance is invalid");
  }
  for (double value : episode.terminal_returns) {
    if (!std::isfinite(value)) return fail("terminal return is nonfinite");
  }
  VrpoSeatValues reward_sums = {0.0, 0.0, 0.0, 0.0};
  for (size_t t = 0; t < episode.rows.size(); ++t) {
    const VrpoCapturedRow& row = episode.rows[t];
    std::string row_error;
    if (!vrpo_capture_internal::ValidateRowCore(
            episode, row, t, &row_error)) {
      return fail(row_error);
    }
    if (row.terminal_after != (t + 1 == episode.rows.size())) {
      return fail("captured terminal boundary is not exactly the final row");
    }
    if (row.row_sha256 != vrpo_capture_internal::RowSha256(row)) {
      return fail("captured row hash mismatch");
    }
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      const double expected =
          (t + 1 == episode.rows.size())
              ? std::clamp(episode.terminal_returns[seat] / 4.0, -1.0, 1.0)
              : 0.0;
      if (row.rewards[seat] != expected) {
        return fail("captured zero-shaping reward placement is wrong");
      }
      reward_sums[seat] += row.rewards[seat];
    }
  }
  for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
    const double expected =
        std::clamp(episode.terminal_returns[seat] / 4.0, -1.0, 1.0);
    if (reward_sums[seat] != expected) {
      return fail("captured per-seat reward conservation failed");
    }
  }
  if (episode.capture_sha256 !=
      vrpo_capture_internal::EpisodeSha256(episode)) {
    return fail("captured episode hash mismatch");
  }
  return true;
}

inline bool FinalizeVrpoZeroShapingEpisode(
    VrpoCapturedEpisode* episode, const VrpoSeatValues& terminal_returns,
    const VrpoZeroShapingRewardConfig& config, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (episode == nullptr) return fail("null captured episode");
  if (episode->rows.empty()) return fail("captured episode has no rows");
  if (config.reward_scale != 4.0 || !std::isfinite(config.gamma) ||
      config.gamma < 0.0 || config.gamma > 1.0 ||
      !std::isfinite(config.lambda) || config.lambda < 0.0 ||
      config.lambda > 1.0 ||
      !std::isfinite(config.probability_tolerance) ||
      config.probability_tolerance < 0.0 ||
      config.probability_tolerance > kVrpoMaxProbabilityTolerance) {
    return fail("zero-shaping reward config is invalid");
  }
  for (double coefficient :
       {config.shaped_reward_weight, config.tleilaxu_breadcrumb_weight,
        config.tleilaxu_level7_breadcrumb_weight,
        config.specimen_exchange_penalty}) {
    if (!std::isfinite(coefficient) || coefficient != 0.0) {
      return fail("all shaping coefficients must be exactly zero");
    }
  }
  for (double value : terminal_returns) {
    if (!std::isfinite(value)) return fail("terminal return is nonfinite");
  }

  VrpoCapturedEpisode candidate = *episode;
  candidate.capture_schema_sha256 = kVrpoCaptureSchemaSha256;
  candidate.reward_convention_sha256 =
      kVrpoZeroShapingRewardConventionSha256;
  candidate.reward_scale = config.reward_scale;
  candidate.gamma = config.gamma;
  candidate.lambda = config.lambda;
  candidate.probability_tolerance = config.probability_tolerance;
  candidate.terminal_returns = terminal_returns;
  candidate.rewards_finalized = true;
  for (size_t t = 0; t < candidate.rows.size(); ++t) {
    std::string row_error;
    if (!vrpo_capture_internal::ValidateRowCore(
            candidate, candidate.rows[t], t, &row_error)) {
      return fail(row_error);
    }
    candidate.rows[t].rewards = {0.0, 0.0, 0.0, 0.0};
    candidate.rows[t].terminal_after = false;
  }
  candidate.rows.back().terminal_after = true;
  for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
    candidate.rows.back().rewards[seat] =
        std::clamp(terminal_returns[seat] / 4.0, -1.0, 1.0);
  }
  vrpo_capture_internal::RefreshHashes(&candidate);
  std::string validation_error;
  if (!ValidateVrpoCapturedEpisode(candidate, &validation_error)) {
    return fail("finalized capture validation failed: " + validation_error);
  }
  *episode = std::move(candidate);
  return true;
}

inline bool VrpoCapturedEpisodeToTimeline(
    const VrpoCapturedEpisode& episode,
    const std::vector<std::vector<VrpoActorRelativeSeatValues>>&
        relative_legal_q_by_row,
    std::vector<VrpoTimelineRow>* timeline, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (timeline != nullptr) timeline->clear();
    return false;
  };
  if (timeline == nullptr) return fail("null timeline output");
  timeline->clear();
  std::string validation_error;
  if (!ValidateVrpoCapturedEpisode(episode, &validation_error)) {
    return fail("captured episode is invalid: " + validation_error);
  }
  if (relative_legal_q_by_row.size() != episode.rows.size()) {
    return fail("relative Q outer row count differs from capture");
  }
  std::vector<VrpoTimelineRow> converted;
  converted.reserve(episode.rows.size());
  for (size_t t = 0; t < episode.rows.size(); ++t) {
    const VrpoCapturedRow& captured = episode.rows[t];
    if (relative_legal_q_by_row[t].size() !=
        captured.legal_actions.size()) {
      return fail("relative Q legal width differs from capture");
    }
    std::vector<VrpoSeatValues> absolute_q;
    std::string adapter_error;
    if (!VrpoActorRelativeQToAbsolute(
            captured.actor, relative_legal_q_by_row[t], &absolute_q,
            &adapter_error)) {
      return fail("relative Q conversion failed: " + adapter_error);
    }
    VrpoTimelineRow row;
    row.episode_id = captured.episode_id;
    row.actor = captured.actor;
    row.legal_actions = captured.legal_actions;
    row.chosen_index = captured.chosen_index;
    row.chosen_action = captured.chosen_action;
    row.legal_probabilities = captured.legal_behavior_probabilities;
    row.legal_q_values = std::move(absolute_q);
    row.rewards = captured.rewards;
    row.terminal_after = captured.terminal_after;
    converted.push_back(std::move(row));
  }
  *timeline = std::move(converted);
  return true;
}

inline bool ComputeVrpoCapturedEpisodeReference(
    const VrpoCapturedEpisode& episode,
    const std::vector<std::vector<VrpoActorRelativeSeatValues>>&
        relative_legal_q_by_row,
    VrpoReferenceTrace* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoReferenceTrace{};
    return false;
  };
  if (output == nullptr) return fail("null captured-reference output");
  *output = VrpoReferenceTrace{};
  std::vector<VrpoTimelineRow> timeline;
  std::string conversion_error;
  if (!VrpoCapturedEpisodeToTimeline(
          episode, relative_legal_q_by_row, &timeline,
          &conversion_error)) {
    return fail(conversion_error);
  }
  std::string reference_error;
  if (!ComputeVrpoExpectedSarsaLambdaReference(
          timeline, episode.gamma, episode.lambda, output,
          &reference_error, episode.probability_tolerance)) {
    return fail("reference trace failed: " + reference_error);
  }
  return true;
}

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_H_
