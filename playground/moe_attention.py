import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from playground.Sublayers import MultiHeadAttention
import numpy as np
from torch.nn.init import xavier_uniform_


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        self._part_sizes = list((gates > 0).sum(0).numpy())
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        # quick hack
        self._nonzero_gates = self._nonzero_gates.view(-1, 1, 1)

    def dispatch(self, experts, q, k, v, mask):
        query_exp = q[self._batch_index, :, :]
        key_exp = k[self._batch_index, :, :]
        value_exp = v[self._batch_index, :, :]

        queries_exp = torch.split(query_exp, self._part_sizes, dim=0)
        keys_exp = torch.split(key_exp, self._part_sizes, dim=0)
        values_exp = torch.split(value_exp, self._part_sizes, dim=0)

        expert_outputs = []

        for i in range(len(experts)):

            if queries_exp[i].size(0) == 0:
                attn_zeros_out = torch.zeros(0, queries_exp[i].size(1), queries_exp[i].size(2), requires_grad=True)
                expert_outputs.append(attn_zeros_out)

            else:
                expert_outputs.append(experts[i](queries_exp[i], keys_exp[i], values_exp[i], mask))

        return expert_outputs

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)

        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), requires_grad=True)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoeMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, num_experts, noisy_gating=True, k=2):
        super(MoeMultiHeadAttention, self).__init__()
        # initialise some components of moe only if noisy_gating is activated
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.k = k
        self.experts = nn.ModuleList(
            MultiHeadAttention(heads=num_heads, d_model=embed_dim, dropout=dropout) for _ in range(self.num_experts))

        self.w_gate = nn.Parameter(torch.Tensor(embed_dim, num_experts))
        self.w_noise = nn.Parameter(torch.Tensor(embed_dim, num_experts))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        assert (self.k <= self.num_experts)

        self.reset_parameters()

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        x[x != x] = float(0.0)

        clean_logits = torch.tensor((), dtype=torch.float)
        clean_logits = clean_logits.new_zeros((x.size(0), self.num_experts))
        raw_noise_stddev = torch.tensor((), dtype=torch.float)
        raw_noise_stddev = raw_noise_stddev.new_zeros((x.size(0), self.num_experts))

        for i in range(x.size(1)):
            word_clone = x[:, i, :].clone()
            clean_logits = clean_logits + word_clone @ self.w_gate

            if self.noisy_gating:
                raw_noise_stddev = raw_noise_stddev + word_clone @ self.w_noise

        if self.noisy_gating:
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k].clone()
        top_k_indices = top_indices[:, :self.k].clone()
        top_k_gates = self.softmax(top_k_logits)

        top_k_gates = top_k_gates + float(1e-9)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, q, k, v, mask,
                train=True, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(q, train)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef  # can the loss coefficient be trainable?

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_outputs = dispatcher.dispatch(experts=self.experts, q=q, k=k, v=v, mask=mask)
        gates = dispatcher.expert_to_gates()
        attn_output = dispatcher.combine(expert_outputs)
        return attn_output, loss
