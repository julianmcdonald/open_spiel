#ifndef OPEN_SPIEL_EXAMPLES_DUNE_VRPO_H_
#define OPEN_SPIEL_EXAMPLES_DUNE_VRPO_H_

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dune_sha256.h"
#include "open_spiel/games/dune_imperium/dune_imperium.h"
#include "open_spiel/spiel.h"
#include "open_spiel/utils/json.h"

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
inline constexpr int kVrpoQPreflightMaxChunkRows = 256;
inline constexpr double kVrpoQPreflightMaxAgreementTolerance = 1e-6;
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
inline std::atomic<int64_t> g_vrpo_q_constructor_calls{0};
inline std::atomic<int64_t> g_vrpo_q_forward_checked_calls{0};

inline void ResetVrpoQInstrumentation() {
  g_vrpo_q_constructor_calls.store(0, std::memory_order_relaxed);
  g_vrpo_q_forward_checked_calls.store(0, std::memory_order_relaxed);
}

inline int64_t VrpoQConstructorCalls() {
  return g_vrpo_q_constructor_calls.load(std::memory_order_relaxed);
}

inline int64_t VrpoQForwardCheckedCalls() {
  return g_vrpo_q_forward_checked_calls.load(std::memory_order_relaxed);
}

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
    g_vrpo_q_constructor_calls.fetch_add(1, std::memory_order_relaxed);
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
    g_vrpo_q_forward_checked_calls.fetch_add(1, std::memory_order_relaxed);
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

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

struct VrpoTensorReferenceTrace {
  uint64_t episode_id = 0;
  std::vector<VrpoReferenceRow> rows;
  std::string canonical_sha256;
};

inline bool ComputeVrpoExpectedSarsaLambdaTensorReference(
    const std::vector<VrpoTimelineRow>& timeline, double gamma,
    double lambda, VrpoTensorReferenceTrace* output, std::string* error,
    double probability_tolerance = 1e-9) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoTensorReferenceTrace{};
    return false;
  };
  if (output == nullptr) return fail("null tensor-reference output");
  *output = VrpoTensorReferenceTrace{};
  if (timeline.empty()) return fail("tensor-reference timeline is empty");
  if (!std::isfinite(gamma) || gamma < 0.0 || gamma > 1.0 ||
      !std::isfinite(lambda) || lambda < 0.0 || lambda > 1.0 ||
      !std::isfinite(probability_tolerance) ||
      probability_tolerance < 0.0 ||
      probability_tolerance > kVrpoMaxProbabilityTolerance) {
    return fail("tensor-reference gamma/lambda/tolerance is invalid");
  }

  const uint64_t episode_id = timeline.front().episode_id;
  const int64_t row_count = static_cast<int64_t>(timeline.size());
  const auto options = torch::TensorOptions()
                           .dtype(torch::kFloat64)
                           .device(torch::kCPU);
  torch::Tensor v = torch::zeros({row_count, kVrpoNumSeats}, options);
  torch::Tensor chosen_q = torch::zeros({row_count, kVrpoNumSeats}, options);
  torch::Tensor rewards = torch::zeros({row_count, kVrpoNumSeats}, options);
  torch::Tensor actors = torch::empty(
      {row_count}, torch::TensorOptions().dtype(torch::kInt64));

  for (int64_t t = 0; t < row_count; ++t) {
    const VrpoTimelineRow& row = timeline[t];
    if (row.episode_id != episode_id) {
      return fail("tensor-reference crosses an episode boundary");
    }
    if (row.actor < 0 || row.actor >= kVrpoNumSeats) {
      return fail("tensor-reference actor is out of range");
    }
    if (row.terminal_after != (t + 1 == row_count)) {
      return fail("tensor-reference terminal boundary is invalid");
    }
    const size_t legal_count = row.legal_actions.size();
    if (legal_count == 0 || row.legal_probabilities.size() != legal_count ||
        row.legal_q_values.size() != legal_count || row.chosen_index < 0 ||
        row.chosen_index >= static_cast<int>(legal_count) ||
        row.legal_actions[row.chosen_index] != row.chosen_action) {
      return fail("tensor-reference row widths/choice are invalid");
    }
    std::set<Action> seen;
    double probability_sum = 0.0;
    std::vector<double> q_flat;
    q_flat.reserve(legal_count * kVrpoNumSeats);
    for (size_t a = 0; a < legal_count; ++a) {
      const Action action = row.legal_actions[a];
      if (action < 0 || action >= kVrpoDuneActionDim ||
          !seen.insert(action).second) {
        return fail("tensor-reference legal action is invalid");
      }
      const double probability = row.legal_probabilities[a];
      if (!std::isfinite(probability) || probability < 0.0 ||
          probability > 1.0 + probability_tolerance) {
        return fail("tensor-reference probability is invalid");
      }
      probability_sum += probability;
      for (double q : row.legal_q_values[a]) {
        if (!std::isfinite(q)) {
          return fail("tensor-reference Q input is nonfinite");
        }
        q_flat.push_back(q);
      }
    }
    if (!std::isfinite(probability_sum) ||
        std::abs(probability_sum - 1.0) > probability_tolerance) {
      return fail("tensor-reference probabilities do not sum to one");
    }
    for (double reward : row.rewards) {
      if (!std::isfinite(reward)) {
        return fail("tensor-reference reward is nonfinite");
      }
    }

    torch::Tensor probabilities = torch::from_blob(
        const_cast<double*>(row.legal_probabilities.data()),
        {static_cast<int64_t>(legal_count)}, options).clone();
    torch::Tensor q = torch::from_blob(
        q_flat.data(),
        {static_cast<int64_t>(legal_count), kVrpoNumSeats}, options).clone();
    v[t].copy_((probabilities.unsqueeze(1) * q).sum(0));
    chosen_q[t].copy_(q[row.chosen_index]);
    rewards[t].copy_(torch::from_blob(
        const_cast<double*>(row.rewards.data()), {kVrpoNumSeats}, options));
    actors[t] = static_cast<int64_t>(row.actor);
  }

  torch::Tensor next_v = torch::zeros_like(v);
  if (row_count > 1) {
    next_v.narrow(0, 0, row_count - 1)
        .copy_(v.narrow(0, 1, row_count - 1));
  }
  torch::Tensor delta = rewards + gamma * next_v - chosen_q;
  torch::Tensor g = torch::zeros_like(delta);
  torch::Tensor next_g = torch::zeros({kVrpoNumSeats}, options);
  for (int64_t reverse = row_count; reverse-- > 0;) {
    g[reverse].copy_(delta[reverse] + gamma * lambda * next_g);
    next_g.copy_(g[reverse]);
  }
  torch::Tensor q_target = chosen_q + g;
  torch::Tensor actor_advantage =
      (chosen_q - v + g).gather(1, actors.unsqueeze(1)).squeeze(1);
  for (const torch::Tensor& tensor :
       {v, delta, g, q_target, actor_advantage}) {
    if (!torch::isfinite(tensor).all().item<bool>()) {
      return fail("tensor-reference derived output is nonfinite");
    }
  }

  VrpoTensorReferenceTrace result;
  result.episode_id = episode_id;
  result.rows.resize(timeline.size());
  std::string payload = "dune_vrpo_tensor_expected_sarsa_lambda_v1";
  payload.append(reinterpret_cast<const char*>(&gamma), sizeof(gamma));
  payload.append(reinterpret_cast<const char*>(&lambda), sizeof(lambda));
  const uint64_t count = timeline.size();
  payload.append(reinterpret_cast<const char*>(&count), sizeof(count));
  for (int64_t t = 0; t < row_count; ++t) {
    VrpoReferenceRow& row = result.rows[t];
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      row.v[seat] = v[t][seat].item<double>();
      row.delta[seat] = delta[t][seat].item<double>();
      row.g[seat] = g[t][seat].item<double>();
      row.q_target[seat] = q_target[t][seat].item<double>();
    }
    row.actor_advantage = actor_advantage[t].item<double>();
    for (double value : row.v) {
      payload.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    for (double value : row.delta) {
      payload.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    for (double value : row.g) {
      payload.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    for (double value : row.q_target) {
      payload.append(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    payload.append(reinterpret_cast<const char*>(&row.actor_advantage),
                   sizeof(row.actor_advantage));
  }
  result.canonical_sha256 = ComputeStringSHA256(payload);
  *output = std::move(result);
  return true;
}

struct VrpoReferenceAgreement {
  int64_t compared_values = 0;
  int64_t mismatch_count = 0;
  double max_abs_v = 0.0;
  double max_abs_delta = 0.0;
  double max_abs_g = 0.0;
  double max_abs_q_target = 0.0;
  double max_abs_actor_advantage = 0.0;
};

inline bool CompareVrpoReferenceTraces(
    const VrpoReferenceTrace& scalar,
    const VrpoTensorReferenceTrace& tensor, double abs_tolerance,
    double rel_tolerance, VrpoReferenceAgreement* output,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoReferenceAgreement{};
    return false;
  };
  if (output == nullptr) return fail("null reference-agreement output");
  *output = VrpoReferenceAgreement{};
  if (scalar.episode_id != tensor.episode_id ||
      scalar.rows.size() != tensor.rows.size() || scalar.rows.empty()) {
    return fail("reference-agreement row identity differs");
  }
  for (double tolerance : {abs_tolerance, rel_tolerance}) {
    if (!std::isfinite(tolerance) || tolerance < 0.0 ||
        tolerance > kVrpoQPreflightMaxAgreementTolerance) {
      return fail("reference-agreement tolerance is invalid");
    }
  }
  auto compare = [&](double first, double second, double* maximum) {
    const double difference = std::abs(first - second);
    *maximum = std::max(*maximum, difference);
    ++output->compared_values;
    const double bound = abs_tolerance +
        rel_tolerance * std::max(std::abs(first), std::abs(second));
    if (!std::isfinite(first) || !std::isfinite(second) ||
        difference > bound) {
      ++output->mismatch_count;
    }
  };
  for (size_t t = 0; t < scalar.rows.size(); ++t) {
    for (int seat = 0; seat < kVrpoNumSeats; ++seat) {
      compare(scalar.rows[t].v[seat], tensor.rows[t].v[seat],
              &output->max_abs_v);
      compare(scalar.rows[t].delta[seat], tensor.rows[t].delta[seat],
              &output->max_abs_delta);
      compare(scalar.rows[t].g[seat], tensor.rows[t].g[seat],
              &output->max_abs_g);
      compare(scalar.rows[t].q_target[seat],
              tensor.rows[t].q_target[seat],
              &output->max_abs_q_target);
    }
    compare(scalar.rows[t].actor_advantage,
            tensor.rows[t].actor_advantage,
            &output->max_abs_actor_advantage);
  }
  if (output->mismatch_count != 0) {
    if (error != nullptr) {
      *error = "scalar/tensor reference recurrence differs";
    }
    return false;
  }
  return true;
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

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

inline bool ValidateVrpoCapturedEpisode(const VrpoCapturedEpisode& episode,
                                        std::string* error);

class VrpoCapturedEpisodeBuffer {
 public:
  bool PublishValidated(VrpoCapturedEpisode episode, std::string* error) {
    std::string validation_error;
    if (!ValidateVrpoCapturedEpisode(episode, &validation_error)) {
      if (error != nullptr) {
        *error = "refusing malformed VRPO episode publication: " +
                 validation_error;
      }
      return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!episode_ids_.insert(episode.episode_id).second) {
      if (error != nullptr) *error = "duplicate VRPO episode ID";
      return false;
    }
    episodes_.push_back(std::move(episode));
    return true;
  }

  std::vector<VrpoCapturedEpisode> TakeSorted() {
    std::lock_guard<std::mutex> lock(mu_);
    std::sort(episodes_.begin(), episodes_.end(),
              [](const auto& first, const auto& second) {
                return first.episode_id < second.episode_id;
              });
    std::vector<VrpoCapturedEpisode> result = std::move(episodes_);
    episodes_.clear();
    episode_ids_.clear();
    return result;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return episodes_.size();
  }

 private:
  mutable std::mutex mu_;
  std::set<uint64_t> episode_ids_;
  std::vector<VrpoCapturedEpisode> episodes_;
};

struct VrpoCaptureStartupConfig {
  std::string game;
  std::string registration_id;
  std::string source_root;
  std::string source_sha256;
  bool diagnostics_only = false;
  std::string init_mode;
  int rollout_games = 0;
  int threads = 0;
  bool rollout_amp = true;
  bool train_amp = true;
  bool allow_tf32 = true;
  bool pipeline = false;
  bool online_search_collection = false;
  bool search_pi_mode = false;
  bool train_value_only = false;
  bool sample_counterfactual_states = false;
  bool has_search_label_dir = false;
  double shaped_reward_weight = 0.0;
  double tleilaxu_breadcrumb_weight = 0.0;
  double tleilaxu_level7_breadcrumb_weight = 0.0;
  double specimen_exchange_penalty = 0.0;
  double reward_scale = 4.0;
  double gamma = 1.0;
  double lambda = 1.0;
};

inline bool ValidateVrpoCaptureStartupConfig(
    const VrpoCaptureStartupConfig& config, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  auto lower_hex64 = [](const std::string& value) {
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](char c) {
             return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
  };
  if (config.game != "dune_imperium") return fail("VRPO capture requires exact Dune game");
  if (!config.diagnostics_only || config.init_mode != "diagnostic") {
    return fail("VRPO capture requires diagnostics_only diagnostic init");
  }
  if (config.registration_id.empty() || config.source_root.empty() ||
      !lower_hex64(config.source_sha256)) {
    return fail("VRPO capture registration/source identity is invalid");
  }
  if (config.rollout_games < 1 || config.rollout_games > 4 ||
      config.threads < 1 || config.threads > 4) {
    return fail("VRPO capture games/threads exceed registered ceilings");
  }
  if (config.rollout_amp || config.train_amp || !config.allow_tf32) {
    return fail("VRPO capture requires FP32 rollout/learner and TF32 enabled");
  }
  if (config.pipeline || config.online_search_collection ||
      config.search_pi_mode || config.train_value_only ||
      config.sample_counterfactual_states || config.has_search_label_dir) {
    return fail("VRPO capture forbids training/search/counterfactual paths");
  }
  for (double coefficient :
       {config.shaped_reward_weight, config.tleilaxu_breadcrumb_weight,
        config.tleilaxu_level7_breadcrumb_weight,
        config.specimen_exchange_penalty}) {
    if (!std::isfinite(coefficient) || coefficient != 0.0) {
      return fail("VRPO capture requires all shaping coefficients exactly zero");
    }
  }
  if (config.reward_scale != 4.0 || !std::isfinite(config.gamma) ||
      config.gamma < 0.0 || config.gamma > 1.0 ||
      !std::isfinite(config.lambda) || config.lambda < 0.0 ||
      config.lambda > 1.0) {
    return fail("VRPO capture reward scale/gamma/lambda is invalid");
  }
  return true;
}

inline bool VrpoCaptureShouldConstructOptimizer(bool capture_active) {
  return !capture_active;
}

struct VrpoQPreflightStartupConfig {
  VrpoCaptureStartupConfig capture;
  uint64_t q_init_seed = 0;
  int q_chunk_rows = 0;
  double agreement_abs_tolerance = 1e-10;
  double agreement_rel_tolerance = 1e-10;
  int64_t gpu_peak_increment_limit_bytes = 0;
};

inline bool ValidateVrpoQPreflightStartupConfig(
    const VrpoQPreflightStartupConfig& config, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  std::string capture_error;
  if (!ValidateVrpoCaptureStartupConfig(config.capture, &capture_error)) {
    return fail("Q preflight capture contract failed: " + capture_error);
  }
  if (config.capture.rollout_games != 1 || config.capture.threads != 1) {
    return fail("Q preflight requires exactly one game and one thread");
  }
  if (config.q_init_seed == 0 || config.q_chunk_rows < 1 ||
      config.q_chunk_rows > kVrpoQPreflightMaxChunkRows) {
    return fail("Q preflight seed/chunk contract is invalid");
  }
  for (double tolerance : {config.agreement_abs_tolerance,
                           config.agreement_rel_tolerance}) {
    if (!std::isfinite(tolerance) || tolerance < 0.0 ||
        tolerance > kVrpoQPreflightMaxAgreementTolerance) {
      return fail("Q preflight agreement tolerance is invalid");
    }
  }
  if (config.gpu_peak_increment_limit_bytes <= 0) {
    return fail("Q preflight GPU memory ceiling must be positive");
  }
  return true;
}

inline bool BuildVrpoChunkRanges(
    size_t rows, int chunk_rows,
    std::vector<std::pair<size_t, size_t>>* ranges, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (ranges != nullptr) ranges->clear();
    return false;
  };
  if (ranges == nullptr) return fail("null Q chunk-range output");
  ranges->clear();
  if (rows == 0 || chunk_rows < 1 ||
      chunk_rows > kVrpoQPreflightMaxChunkRows) {
    return fail("Q chunk range input is invalid");
  }
  for (size_t begin = 0; begin < rows;
       begin += static_cast<size_t>(chunk_rows)) {
    const size_t count = std::min(
        static_cast<size_t>(chunk_rows), rows - begin);
    ranges->push_back({begin, count});
  }
  size_t cursor = 0;
  for (const auto& range : *ranges) {
    if (range.first != cursor || range.second == 0 ||
        range.second > static_cast<size_t>(chunk_rows)) {
      return fail("Q chunk ranges are not an exact ordered partition");
    }
    cursor += range.second;
  }
  if (cursor != rows) return fail("Q chunk ranges do not cover every row");
  return true;
}

struct VrpoRolloutPairingView {
  uint64_t episode_id = 0;
  Player actor = kInvalidPlayer;
  const std::vector<float>* actor_observation = nullptr;
  const std::vector<Action>* legal_actions = nullptr;
  Action action = kInvalidAction;
  float chosen_log_probability = 0.0f;
};

inline bool ValidateVrpoCaptureRolloutPairing(
    const VrpoCapturedRow& captured, const VrpoRolloutPairingView& rollout,
    std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  if (rollout.actor_observation == nullptr || rollout.legal_actions == nullptr) {
    return fail("capture/rollout pairing view is incomplete");
  }
  if (captured.episode_id != rollout.episode_id) {
    return fail("capture/rollout episode mismatch");
  }
  if (captured.actor != rollout.actor) {
    return fail("capture/rollout actor mismatch");
  }
  if (captured.actor_observation.size() != rollout.actor_observation->size() ||
      std::memcmp(captured.actor_observation.data(),
                  rollout.actor_observation->data(),
                  captured.actor_observation.size() * sizeof(float)) != 0) {
    return fail("capture/rollout actor observation mismatch");
  }
  if (captured.legal_actions != *rollout.legal_actions) {
    return fail("capture/rollout ordered legal actions mismatch");
  }
  if (captured.chosen_action != rollout.action) {
    return fail("capture/rollout chosen action mismatch");
  }
  if (captured.chosen_index < 0 ||
      captured.chosen_index >=
          static_cast<int>(captured.legal_behavior_probabilities.size())) {
    return fail("capture chosen index is invalid for pairing");
  }
  const double probability =
      captured.legal_behavior_probabilities[captured.chosen_index];
  const float expected_log = static_cast<float>(std::log(probability));
  if (rollout.chosen_log_probability != expected_log) {
    return fail("capture/rollout chosen log-probability mismatch");
  }
  return true;
}

inline bool ValidateVrpoSortedEpisodeIds(
    const std::vector<VrpoCapturedEpisode>& episodes,
    uint64_t start_episode_id, int rollout_games, std::string* error) {
  if (rollout_games <= 0 ||
      episodes.size() != static_cast<size_t>(rollout_games)) {
    if (error != nullptr) *error = "captured episode count mismatch";
    return false;
  }
  for (int i = 0; i < rollout_games; ++i) {
    if (episodes[i].episode_id != start_episode_id + i) {
      if (error != nullptr) *error = "captured episode IDs are not exact/sorted";
      return false;
    }
  }
  return true;
}

struct VrpoPreQGateResult {
  bool valid = false;
  int64_t captured_rows = 0;
  int64_t rollout_rows = 0;
  int64_t paired_rows = 0;
  std::vector<std::string> errors;
};

inline VrpoPreQGateResult ValidateVrpoPreQCaptureRolloutGate(
    const std::vector<VrpoCapturedEpisode>& episodes,
    const std::vector<VrpoRolloutPairingView>& rollout_rows,
    uint64_t start_episode_id, int rollout_games) {
  VrpoPreQGateResult result;
  result.rollout_rows = static_cast<int64_t>(rollout_rows.size());
  auto reject = [&](const std::string& message) {
    result.errors.push_back(message);
  };
  std::string id_error;
  if (!ValidateVrpoSortedEpisodeIds(
          episodes, start_episode_id, rollout_games, &id_error)) {
    reject("pre-Q exact episode range failed: " + id_error);
  }
  size_t cursor = 0;
  for (const auto& episode : episodes) {
    result.captured_rows += static_cast<int64_t>(episode.rows.size());
    std::string episode_error;
    if (!ValidateVrpoCapturedEpisode(episode, &episode_error)) {
      reject("pre-Q captured episode validation failed: " + episode_error);
      cursor += episode.rows.size();
      continue;
    }
    for (const auto& captured : episode.rows) {
      if (cursor >= rollout_rows.size()) {
        reject("pre-Q rollout ended before captured rows");
        ++cursor;
        continue;
      }
      std::string pairing_error;
      if (ValidateVrpoCaptureRolloutPairing(
              captured, rollout_rows[cursor], &pairing_error)) {
        ++result.paired_rows;
      } else {
        reject("pre-Q capture/rollout pairing failed at row " +
               std::to_string(cursor) + ": " + pairing_error);
      }
      ++cursor;
    }
  }
  if (cursor != rollout_rows.size() ||
      result.captured_rows != result.rollout_rows) {
    reject("pre-Q capture/rollout row counts differ");
  }
  if (result.paired_rows != result.captured_rows) {
    reject("pre-Q exhaustive row pairing is incomplete");
  }
  result.valid = result.errors.empty();
  return result;
}

inline bool ValidateVrpoCaptureRewardMetadata(
    const VrpoCapturedEpisode& episode, double registered_reward_scale,
    double registered_gamma, double registered_lambda,
    double registered_probability_tolerance, std::string* error) {
  const bool valid = episode.reward_scale == registered_reward_scale &&
      episode.gamma == registered_gamma &&
      episode.lambda == registered_lambda &&
      episode.probability_tolerance == registered_probability_tolerance;
  if (!valid && error != nullptr) {
    *error = "captured reward metadata differs from registered values";
  }
  return valid;
}

inline std::vector<std::string> VrpoCaptureSourceRelativePaths() {
  return {"open_spiel/examples/dune_ppo_train.cc",
          "open_spiel/examples/dune_network.h",
          "open_spiel/examples/dune_vrpo.h",
          "open_spiel/games/dune_imperium/dune_imperium.h",
          "open_spiel/games/dune_imperium/dune_imperium.cc"};
}

inline std::vector<std::string> VrpoQPreflightSourceRelativePaths() {
  std::vector<std::string> paths = VrpoCaptureSourceRelativePaths();
  paths.push_back("open_spiel/examples/dune_seed_utils.h");
  paths.push_back("open_spiel/examples/dune_sha256.h");
  return paths;
}

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

inline bool VrpoLegalQEvidenceSha256(
    const VrpoCapturedEpisode& episode,
    const std::vector<std::vector<VrpoActorRelativeSeatValues>>&
        relative_legal_q_by_row,
    std::string* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null legal-Q evidence hash output");
  output->clear();
  std::vector<VrpoTimelineRow> timeline;
  std::string conversion_error;
  if (!VrpoCapturedEpisodeToTimeline(
          episode, relative_legal_q_by_row, &timeline,
          &conversion_error)) {
    return fail("legal-Q evidence conversion failed: " + conversion_error);
  }
  std::string payload = "dune_vrpo_legal_q_evidence_v1";
  payload.append(episode.capture_sha256);
  for (size_t t = 0; t < timeline.size(); ++t) {
    const VrpoTimelineRow& absolute = timeline[t];
    vrpo_capture_internal::AppendPod(&payload, absolute.episode_id);
    const int64_t row_index = static_cast<int64_t>(t);
    vrpo_capture_internal::AppendPod(&payload, row_index);
    vrpo_capture_internal::AppendPod(&payload, absolute.actor);
    const uint64_t legal_count = absolute.legal_actions.size();
    vrpo_capture_internal::AppendPod(&payload, legal_count);
    for (size_t a = 0; a < legal_count; ++a) {
      vrpo_capture_internal::AppendPod(&payload,
                                       absolute.legal_actions[a]);
      for (double value : relative_legal_q_by_row[t][a].slots) {
        vrpo_capture_internal::AppendPod(&payload, value);
      }
      for (double value : absolute.legal_q_values[a]) {
        vrpo_capture_internal::AppendPod(&payload, value);
      }
    }
  }
  *output = ComputeStringSHA256(payload);
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

// ---------------------------------------------------------------------------
// Phase 4a: pure four-arm schema/bootstrap contracts. Nothing below has a
// production caller; it exists to freeze the experiment before any optimizer
// or training loop is implemented.

enum class VrpoPhase4Algorithm { kPpo, kVrpo };

inline std::string VrpoPhase4AlgorithmName(VrpoPhase4Algorithm algorithm) {
  return algorithm == VrpoPhase4Algorithm::kPpo ? "ppo" : "vrpo";
}

struct VrpoPhase4ArmConfig {
  std::string arm_id;
  VrpoPhase4Algorithm algorithm = VrpoPhase4Algorithm::kPpo;
  double logit_cap = 10.0;
  bool rollout_amp = false;
  bool train_amp = false;
  bool allow_tf32 = true;
  double gamma = 1.0;
  double lambda = 1.0;
  double reward_scale = 4.0;
  double shaped_reward_weight = 0.0;
  double tleilaxu_breadcrumb_weight = 0.0;
  double tleilaxu_level7_breadcrumb_weight = 0.0;
  double specimen_exchange_penalty = 0.0;
  bool complete_game_batches = true;
  int minibatches_per_epoch = 16;
  int actor_epochs = 4;
  int q_epochs = 4;
  int critic_replay_rollouts = 1;
};

inline std::array<VrpoPhase4ArmConfig, 4> CanonicalVrpoPhase4Arms() {
  VrpoPhase4ArmConfig ppo_cap;
  ppo_cap.arm_id = "PPO_CAP10";
  VrpoPhase4ArmConfig ppo_uncapped = ppo_cap;
  ppo_uncapped.arm_id = "PPO_UNCAPPED";
  ppo_uncapped.logit_cap = 0.0;
  VrpoPhase4ArmConfig vrpo_cap = ppo_cap;
  vrpo_cap.arm_id = "VRPO_CAP10";
  vrpo_cap.algorithm = VrpoPhase4Algorithm::kVrpo;
  VrpoPhase4ArmConfig vrpo_uncapped = vrpo_cap;
  vrpo_uncapped.arm_id = "VRPO_UNCAPPED";
  vrpo_uncapped.logit_cap = 0.0;
  return {ppo_cap, ppo_uncapped, vrpo_cap, vrpo_uncapped};
}

struct VrpoOptimizerGroupSpec {
  std::string optimizer_name;
  std::string group_name;
  double learning_rate = 2.5e-4;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-5;
  double weight_decay = 0.0;
  std::vector<std::string> parameter_prefixes;
};

inline std::vector<VrpoOptimizerGroupSpec>
CanonicalVrpoPhase4OptimizerGroups() {
  return {
      {"actor", "actor_policy", 2.5e-4, 0.9, 0.999, 1e-5, 0.0,
       {"policy_head"}},
      {"actor", "actor_trunk_value", 2.5e-4, 0.9, 0.999, 1e-5, 0.0,
       {"input_layer", "res", "value_head"}},
      {"q", "q_critic", 2.5e-4, 0.9, 0.999, 1e-5, 0.0,
       {"input_layer", "res", "q_head"}},
  };
}

inline std::string VrpoOptimizerGroupSpecSha256(
    const std::vector<VrpoOptimizerGroupSpec>& groups) {
  std::string payload = "dune_vrpo_phase4_optimizer_groups_v1";
  const uint64_t count = groups.size();
  vrpo_capture_internal::AppendPod(&payload, count);
  for (const auto& group : groups) {
    payload.append(group.optimizer_name);
    payload.push_back('\0');
    payload.append(group.group_name);
    payload.push_back('\0');
    vrpo_capture_internal::AppendPod(&payload, group.learning_rate);
    vrpo_capture_internal::AppendPod(&payload, group.beta1);
    vrpo_capture_internal::AppendPod(&payload, group.beta2);
    vrpo_capture_internal::AppendPod(&payload, group.epsilon);
    vrpo_capture_internal::AppendPod(&payload, group.weight_decay);
    const uint64_t prefixes = group.parameter_prefixes.size();
    vrpo_capture_internal::AppendPod(&payload, prefixes);
    for (const auto& prefix : group.parameter_prefixes) {
      payload.append(prefix);
      payload.push_back('\0');
    }
  }
  return ComputeStringSHA256(payload);
}

struct VrpoPhase4ManifestBinding {
  std::string source_actor_model_sha256;
  std::string source_actor_manifest_sha256;
  std::string source_code_sha256;
  int64_t actor_observation_dim = 5580;
  int64_t actor_hidden_dim = 2048;
  int64_t actor_action_dim = 2391;
  int64_t actor_residual_blocks = 8;
  std::string actor_subset_sha256;
  std::string actor_names_shapes_sha256;
  std::string q_init_sha256;
  std::string q_names_shapes_sha256;
  std::string module_layout_sha256;
  std::string optimizer_zero_state_sha256;
  uint64_t q_init_seed = 0;
  std::string experiment_uuid;
  uint64_t base_seed = 0;
  uint64_t start_episode_id = 0;
  uint64_t end_episode_id_inclusive = 0;
};

inline constexpr char kVrpoPhase4ManifestSchemaLabel[] =
    "dune_vrpo_phase4a_manifest_v1_strict_flat";
inline constexpr char kVrpoQAbsoluteBoundaryLabel[] =
    "q_output_actor_relative_then_checked_absolute_seat_0_1_2_3";

inline json::Object VrpoPhase4FingerprintObject(
    const VrpoPhase4ArmConfig& config,
    const VrpoPhase4ManifestBinding& binding) {
  json::Object out;
  out["algorithm"] = VrpoPhase4AlgorithmName(config.algorithm);
  out["logit_cap"] = config.logit_cap;
  out["rollout_amp"] = config.rollout_amp;
  out["train_amp"] = config.train_amp;
  out["allow_tf32"] = config.allow_tf32;
  out["gamma"] = config.gamma;
  out["lambda"] = config.lambda;
  out["reward_scale"] = config.reward_scale;
  out["shaped_reward_weight"] = config.shaped_reward_weight;
  out["tleilaxu_breadcrumb_weight"] = config.tleilaxu_breadcrumb_weight;
  out["tleilaxu_level7_breadcrumb_weight"] =
      config.tleilaxu_level7_breadcrumb_weight;
  out["specimen_exchange_penalty"] = config.specimen_exchange_penalty;
  out["complete_game_batches"] = config.complete_game_batches;
  out["minibatches_per_epoch"] =
      static_cast<int64_t>(config.minibatches_per_epoch);
  out["actor_epochs"] = static_cast<int64_t>(config.actor_epochs);
  out["q_epochs"] = static_cast<int64_t>(config.q_epochs);
  out["critic_replay_rollouts"] =
      static_cast<int64_t>(config.critic_replay_rollouts);
  out["source_actor_model_sha256"] = binding.source_actor_model_sha256;
  out["source_actor_manifest_sha256"] = binding.source_actor_manifest_sha256;
  out["source_code_sha256"] = binding.source_code_sha256;
  out["actor_subset_sha256"] = binding.actor_subset_sha256;
  out["actor_names_shapes_sha256"] = binding.actor_names_shapes_sha256;
  out["central_schema_sha256"] =
      dune_imperium::kVrpoCentralCriticTensorSchemaSha256;
  out["central_approximation_label"] =
      dune_imperium::kVrpoCentralCriticTensorSchemaLabel;
  out["q_init_seed"] = static_cast<int64_t>(binding.q_init_seed);
  out["q_init_sha256"] = binding.q_init_sha256;
  out["q_names_shapes_sha256"] = binding.q_names_shapes_sha256;
  out["q_input_dim"] =
      static_cast<int64_t>(dune_imperium::kVrpoCentralCriticTensorSize);
  out["q_hidden_dim"] = int64_t{kVrpoQHiddenDim};
  out["q_residual_blocks"] = int64_t{kVrpoQNumResBlocks};
  out["q_action_dim"] = int64_t{kVrpoDuneActionDim};
  out["q_reward_perspectives"] = int64_t{kVrpoNumSeats};
  out["q_absolute_boundary_label"] = kVrpoQAbsoluteBoundaryLabel;
  out["reward_convention_sha256"] =
      kVrpoZeroShapingRewardConventionSha256;
  out["module_layout_sha256"] = binding.module_layout_sha256;
  out["optimizer_groups_sha256"] = VrpoOptimizerGroupSpecSha256(
      CanonicalVrpoPhase4OptimizerGroups());
  const auto optimizer_groups = CanonicalVrpoPhase4OptimizerGroups();
  out["optimizer_group_count"] =
      static_cast<int64_t>(optimizer_groups.size());
  for (size_t index = 0; index < optimizer_groups.size(); ++index) {
    const std::string prefix =
        "optimizer_group_" + std::to_string(index) + "_";
    const auto& group = optimizer_groups[index];
    out[prefix + "optimizer"] = group.optimizer_name;
    out[prefix + "name"] = group.group_name;
    out[prefix + "learning_rate"] = group.learning_rate;
    out[prefix + "beta1"] = group.beta1;
    out[prefix + "beta2"] = group.beta2;
    out[prefix + "epsilon"] = group.epsilon;
    out[prefix + "weight_decay"] = group.weight_decay;
    std::string prefixes;
    for (const auto& value : group.parameter_prefixes) {
      if (!prefixes.empty()) prefixes.push_back(',');
      prefixes.append(value);
    }
    out[prefix + "parameter_prefixes"] = prefixes;
  }
  out["optimizer_zero_state_sha256"] =
      binding.optimizer_zero_state_sha256;
  out["optimizer_zero_state_schema"] =
      "adamw_step0_exp_avg0_exp_avg_sq0_all_parameters_v1";
  out["source_optimizer_moments_loaded"] = false;
  out["actor_optimizer_fresh"] = true;
  out["q_optimizer_fresh"] = true;
  out["value_module_present_all_arms"] = true;
  out["q_module_present_all_arms"] = true;
  out["actor_observation_dim"] = binding.actor_observation_dim;
  out["actor_hidden_dim"] = binding.actor_hidden_dim;
  out["actor_action_dim"] = binding.actor_action_dim;
  out["actor_residual_blocks"] = binding.actor_residual_blocks;
  out["experiment_uuid"] = binding.experiment_uuid;
  out["base_seed"] = static_cast<int64_t>(binding.base_seed);
  out["start_episode_id"] =
      static_cast<int64_t>(binding.start_episode_id);
  out["end_episode_id_inclusive"] =
      static_cast<int64_t>(binding.end_episode_id_inclusive);
  return out;
}

inline std::string VrpoPhase4ConfigFingerprint(
    const VrpoPhase4ArmConfig& config,
    const VrpoPhase4ManifestBinding& binding) {
  return ComputeStringSHA256(
      json::ToString(VrpoPhase4FingerprintObject(config, binding)));
}

inline json::Object BuildVrpoPhase4Manifest(
    const VrpoPhase4ArmConfig& config,
    const VrpoPhase4ManifestBinding& binding) {
  json::Object out = VrpoPhase4FingerprintObject(config, binding);
  out["schema"] = kVrpoPhase4ManifestSchemaLabel;
  out["arm_id"] = config.arm_id;
  out["config_fingerprint"] =
      VrpoPhase4ConfigFingerprint(config, binding);
  return out;
}

inline bool ValidateVrpoPhase4ArmConfig(
    const VrpoPhase4ArmConfig& config, std::string* error) {
  const auto canonical = CanonicalVrpoPhase4Arms();
  for (const auto& expected : canonical) {
    if (config.arm_id != expected.arm_id) continue;
    const json::Object actual_object = VrpoPhase4FingerprintObject(
        config, VrpoPhase4ManifestBinding{});
    const json::Object expected_object = VrpoPhase4FingerprintObject(
        expected, VrpoPhase4ManifestBinding{});
    if (json::ToString(actual_object) != json::ToString(expected_object)) {
      if (error != nullptr) {
        *error = "phase4 arm differs outside registered algorithm/cap cell";
      }
      return false;
    }
    return true;
  }
  if (error != nullptr) *error = "phase4 arm ID is not registered";
  return false;
}

inline bool ValidateVrpoPhase4ManifestBinding(
    const VrpoPhase4ManifestBinding& binding, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  auto lower_hex64 = [](const std::string& value) {
    return value.size() == 64 &&
        std::all_of(value.begin(), value.end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        });
  };
  for (const auto* digest :
       {&binding.source_actor_model_sha256,
        &binding.source_actor_manifest_sha256,
        &binding.source_code_sha256, &binding.actor_subset_sha256,
        &binding.actor_names_shapes_sha256, &binding.q_init_sha256,
        &binding.q_names_shapes_sha256, &binding.module_layout_sha256,
        &binding.optimizer_zero_state_sha256}) {
    if (!lower_hex64(*digest)) {
      return fail("phase4 binding contains an invalid SHA-256");
    }
  }
  if (binding.actor_observation_dim != kVrpoDuneInformationStateSize ||
      binding.actor_hidden_dim != 2048 ||
      binding.actor_action_dim != kVrpoDuneActionDim ||
      binding.actor_residual_blocks != 8) {
    return fail("phase4 binding actor architecture is invalid");
  }
  const std::string& uuid = binding.experiment_uuid;
  if (uuid.size() != 36) return fail("phase4 experiment UUID is invalid");
  for (size_t i = 0; i < uuid.size(); ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (uuid[i] != '-') return fail("phase4 experiment UUID is invalid");
    } else if (!((uuid[i] >= '0' && uuid[i] <= '9') ||
                 (uuid[i] >= 'a' && uuid[i] <= 'f'))) {
      return fail("phase4 experiment UUID is invalid");
    }
  }
  if (binding.q_init_seed == 0 || binding.base_seed == 0 ||
      binding.q_init_seed > static_cast<uint64_t>(
          std::numeric_limits<int64_t>::max()) ||
      binding.base_seed > static_cast<uint64_t>(
          std::numeric_limits<int64_t>::max()) ||
      binding.end_episode_id_inclusive < binding.start_episode_id ||
      binding.end_episode_id_inclusive > static_cast<uint64_t>(
          std::numeric_limits<int64_t>::max())) {
    return fail("phase4 binding seed/episode range is invalid");
  }
  return true;
}

inline bool VrpoJsonValuesExactlyEqual(const json::Value& first,
                                       const json::Value& second) {
  if (first.index() != second.index()) return false;
  if (first.IsBool()) return first.GetBool() == second.GetBool();
  if (first.IsInt()) return first.GetInt() == second.GetInt();
  if (first.IsDouble()) return first.GetDouble() == second.GetDouble();
  if (first.IsString()) return first.GetString() == second.GetString();
  return json::ToString(first) == json::ToString(second);
}

inline bool ValidateVrpoPhase4ManifestStrict(
    const json::Object& actual, const VrpoPhase4ArmConfig& config,
    const VrpoPhase4ManifestBinding& binding, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  std::string contract_error;
  if (!ValidateVrpoPhase4ArmConfig(config, &contract_error)) {
    return fail(contract_error);
  }
  if (!ValidateVrpoPhase4ManifestBinding(binding, &contract_error)) {
    return fail(contract_error);
  }
  const json::Object expected = BuildVrpoPhase4Manifest(config, binding);
  if (actual.size() != expected.size()) {
    return fail("phase4 manifest has missing or extra fields");
  }
  for (const auto& item : expected) {
    const auto it = actual.find(item.first);
    if (it == actual.end()) {
      return fail("phase4 manifest missing field: " + item.first);
    }
    if (!VrpoJsonValuesExactlyEqual(it->second, item.second)) {
      return fail("phase4 manifest type/value mismatch: " + item.first);
    }
  }
  return true;
}

inline bool ValidateVrpoPhase4ArmConfigs(
    const std::array<VrpoPhase4ArmConfig, 4>& arms, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  for (size_t i = 0; i < arms.size(); ++i) {
    std::string arm_error;
    if (!ValidateVrpoPhase4ArmConfig(arms[i], &arm_error)) return fail(arm_error);
  }
  std::set<std::string> ids;
  for (const auto& arm : arms) {
    if (!ids.insert(arm.arm_id).second) return fail("phase4 arm IDs repeat");
  }
  return true;
}

inline bool ValidateVrpoPhase4ManifestSetMatched(
    const std::array<json::Object, 4>& manifests, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    return false;
  };
  const std::set<std::string> treatment_fields = {
      "arm_id", "algorithm", "logit_cap", "config_fingerprint"};
  const json::Object& reference = manifests.front();
  for (size_t arm = 1; arm < manifests.size(); ++arm) {
    if (manifests[arm].size() != reference.size()) {
      return fail("phase4 manifests have different field sets");
    }
    for (const auto& item : reference) {
      if (treatment_fields.count(item.first)) continue;
      const auto it = manifests[arm].find(item.first);
      if (it == manifests[arm].end() ||
          !VrpoJsonValuesExactlyEqual(item.second, it->second)) {
        return fail("phase4 matched field differs: " + item.first);
      }
    }
  }
  std::set<std::string> fingerprints;
  std::set<std::string> treatment_cells;
  std::set<std::string> arm_ids;
  for (const auto& manifest : manifests) {
    const auto it = manifest.find("config_fingerprint");
    if (it == manifest.end() || !it->second.IsString() ||
        !fingerprints.insert(it->second.GetString()).second) {
      return fail("phase4 fingerprints are missing or not distinct");
    }
    const auto algorithm = manifest.find("algorithm");
    const auto cap = manifest.find("logit_cap");
    const auto id = manifest.find("arm_id");
    if (algorithm == manifest.end() || !algorithm->second.IsString() ||
        cap == manifest.end() || !cap->second.IsDouble() ||
        id == manifest.end() || !id->second.IsString()) {
      return fail("phase4 treatment identity fields are malformed");
    }
    treatment_cells.insert(
        algorithm->second.GetString() + ":" +
        (cap->second.GetDouble() == 10.0 ? "10" :
         cap->second.GetDouble() == 0.0 ? "0" : "invalid"));
    arm_ids.insert(id->second.GetString());
  }
  if (treatment_cells !=
          std::set<std::string>({"ppo:0", "ppo:10", "vrpo:0", "vrpo:10"}) ||
      arm_ids != std::set<std::string>({"PPO_CAP10", "PPO_UNCAPPED",
                                        "VRPO_CAP10", "VRPO_UNCAPPED"})) {
    return fail("phase4 treatment cells are not the registered 2x2");
  }
  return true;
}

#ifdef OPEN_SPIEL_BUILD_WITH_LIBTORCH

struct VrpoNamedParameterIdentity {
  std::string name;
  std::vector<int64_t> shape;
  std::string value_sha256;
};

inline bool VrpoNamedParameterIdentities(
    torch::nn::Module& module,
    const std::set<std::string>* selected_names,
    std::vector<VrpoNamedParameterIdentity>* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) output->clear();
    return false;
  };
  if (output == nullptr) return fail("null named-parameter identity output");
  output->clear();
  std::set<std::string> found;
  for (const auto& item : module.named_parameters()) {
    if (selected_names != nullptr && !selected_names->count(item.key())) {
      continue;
    }
    torch::Tensor tensor =
        item.value().detach().contiguous().cpu().to(torch::kFloat32);
    if (!torch::isfinite(tensor).all().item<bool>()) {
      return fail("named parameter is nonfinite: " + item.key());
    }
    std::string bytes;
    bytes.append(reinterpret_cast<const char*>(tensor.data_ptr<float>()),
                 tensor.numel() * sizeof(float));
    output->push_back({item.key(),
                       std::vector<int64_t>(tensor.sizes().begin(),
                                            tensor.sizes().end()),
                       ComputeStringSHA256(bytes)});
    found.insert(item.key());
  }
  if (selected_names != nullptr && found != *selected_names) {
    return fail("named actor subset is missing one or more parameters");
  }
  return true;
}

inline std::string VrpoNamedParameterIdentitySha256(
    const std::vector<VrpoNamedParameterIdentity>& identities,
    bool include_values) {
  std::string payload = include_values
      ? "dune_vrpo_named_parameter_values_v1"
      : "dune_vrpo_named_parameter_layout_v1";
  const uint64_t count = identities.size();
  vrpo_capture_internal::AppendPod(&payload, count);
  for (const auto& identity : identities) {
    payload.append(identity.name);
    payload.push_back('\0');
    const uint64_t rank = identity.shape.size();
    vrpo_capture_internal::AppendPod(&payload, rank);
    for (int64_t dim : identity.shape) {
      vrpo_capture_internal::AppendPod(&payload, dim);
    }
    if (include_values) payload.append(identity.value_sha256);
  }
  return ComputeStringSHA256(payload);
}

inline bool CopyVrpoActorSubsetByName(
    torch::nn::Module& source, torch::nn::Module& target,
    const std::set<std::string>& selected_names,
    std::vector<VrpoNamedParameterIdentity>* copied, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (copied != nullptr) copied->clear();
    return false;
  };
  if (copied == nullptr || selected_names.empty()) {
    return fail("actor subset copy requires names and output");
  }
  auto source_parameters = source.named_parameters();
  auto target_parameters = target.named_parameters();
  torch::NoGradGuard no_grad;
  for (const std::string& name : selected_names) {
    auto* source_tensor = source_parameters.find(name);
    auto* target_tensor = target_parameters.find(name);
    if (source_tensor == nullptr || target_tensor == nullptr ||
        source_tensor->sizes() != target_tensor->sizes()) {
      return fail("actor subset name/shape mismatch: " + name);
    }
    target_tensor->copy_(source_tensor->to(target_tensor->device()));
  }
  return VrpoNamedParameterIdentities(
      target, &selected_names, copied, error);
}

struct VrpoPhase4BootIdentity {
  std::string actor_subset_sha256;
  std::string actor_names_shapes_sha256;
  std::string q_init_sha256;
  std::string q_names_shapes_sha256;
  std::string module_layout_sha256;
};

inline bool BuildVrpoPhase4BootIdentity(
    torch::nn::Module& source_actor, torch::nn::Module& target_actor,
    const std::set<std::string>& actor_subset_names, uint64_t q_seed,
    VrpoPhase4BootIdentity* output, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) *error = message;
    if (output != nullptr) *output = VrpoPhase4BootIdentity{};
    return false;
  };
  if (output == nullptr || q_seed == 0) return fail("invalid boot identity input");
  *output = VrpoPhase4BootIdentity{};
  std::vector<VrpoNamedParameterIdentity> actor;
  if (!CopyVrpoActorSubsetByName(
          source_actor, target_actor, actor_subset_names, &actor, error)) {
    return false;
  }
  DuneVrpoQNetImpl q(q_seed);
  std::vector<VrpoNamedParameterIdentity> q_identities;
  if (!VrpoNamedParameterIdentities(
          q, nullptr, &q_identities, error)) {
    return false;
  }
  VrpoPhase4BootIdentity result;
  result.actor_subset_sha256 =
      VrpoNamedParameterIdentitySha256(actor, true);
  result.actor_names_shapes_sha256 =
      VrpoNamedParameterIdentitySha256(actor, false);
  if (!VrpoQModuleParameterSha256(q, &result.q_init_sha256, error)) {
    return false;
  }
  result.q_names_shapes_sha256 =
      VrpoNamedParameterIdentitySha256(q_identities, false);
  std::string layout_payload = "dune_vrpo_phase4_module_layout_v1";
  layout_payload.push_back('\0');
  layout_payload.append(result.actor_names_shapes_sha256);
  layout_payload.append(result.q_names_shapes_sha256);
  result.module_layout_sha256 = ComputeStringSHA256(layout_payload);
  *output = std::move(result);
  return true;
}

inline std::string VrpoOptimizerZeroStateIdentitySha256(
    const std::vector<VrpoOptimizerGroupSpec>& groups,
    const std::vector<VrpoNamedParameterIdentity>& actor,
    const std::vector<VrpoNamedParameterIdentity>& q) {
  std::string payload = "dune_vrpo_phase4_fresh_zero_optimizer_state_v1";
  payload.append(VrpoOptimizerGroupSpecSha256(groups));
  payload.append(VrpoNamedParameterIdentitySha256(actor, false));
  payload.append(VrpoNamedParameterIdentitySha256(q, false));
  const bool source_moments_loaded = false;
  const double zero = 0.0;
  vrpo_capture_internal::AppendPod(&payload, source_moments_loaded);
  for (size_t i = 0; i < actor.size() + q.size(); ++i) {
    vrpo_capture_internal::AppendPod(&payload, zero);  // step
    vrpo_capture_internal::AppendPod(&payload, zero);  // exp_avg identity
    vrpo_capture_internal::AppendPod(&payload, zero);  // exp_avg_sq identity
  }
  return ComputeStringSHA256(payload);
}

#endif  // OPEN_SPIEL_BUILD_WITH_LIBTORCH

}  // namespace open_spiel

#endif  // OPEN_SPIEL_EXAMPLES_DUNE_VRPO_H_
