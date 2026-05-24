#pragma once

#include <torch/torch.h>
#include <memory>

namespace open_spiel {

// 1. Residual Block Definition
struct ResBlockImpl : torch::nn::Module {
  torch::nn::Linear fc1{nullptr};
  torch::nn::Linear fc2{nullptr};
  torch::nn::LayerNorm ln1{nullptr};
  torch::nn::LayerNorm ln2{nullptr};

  ResBlockImpl(int64_t dim) {
    fc1 = register_module("fc1", torch::nn::Linear(dim, dim));
    fc2 = register_module("fc2", torch::nn::Linear(dim, dim));
    ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
  }

  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor residual = x;
    x = torch::relu(ln1->forward(fc1->forward(x)));
    x = ln2->forward(fc2->forward(x));
    return torch::relu(x + residual);
  }
};
TORCH_MODULE(ResBlock);

// 2. Main Dual-Headed Policy-Value Network Definition
struct SharedDunePolicyValueNetImpl : torch::nn::Module {
  torch::nn::Linear input_layer{nullptr};
  std::shared_ptr<ResBlockImpl> res1{nullptr};
  std::shared_ptr<ResBlockImpl> res2{nullptr};
  torch::nn::Linear policy_head{nullptr};
  torch::nn::Linear value_head{nullptr};

  SharedDunePolicyValueNetImpl(int64_t input_dim, int64_t hidden_dim, int64_t action_dim) {
    input_layer = register_module("input_layer", torch::nn::Linear(input_dim, hidden_dim));
    
    // Register the underlying implementation objects wrapped in shared_ptr to solve template deduction
    res1 = register_module("res1", std::make_shared<ResBlockImpl>(hidden_dim));
    res2 = register_module("res2", std::make_shared<ResBlockImpl>(hidden_dim));
    
    policy_head = register_module("policy_head", torch::nn::Linear(hidden_dim, action_dim));
    value_head = register_module("value_head", torch::nn::Linear(hidden_dim, 1));
  }

  struct ModelOutputs {
    torch::Tensor logits;
    torch::Tensor values;
  };

  ModelOutputs forward(torch::Tensor x) {
    x = torch::relu(input_layer->forward(x));
    x = res1->forward(x);
    x = res2->forward(x);
    
    torch::Tensor logits = policy_head->forward(x);
    torch::Tensor values = torch::tanh(value_head->forward(x));
    return {logits, values};
  }
};
TORCH_MODULE(SharedDunePolicyValueNet);

} // namespace open_spiel
