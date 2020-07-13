import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn import functional as F
# from playground import TokenLevelFeedForward # TODO: find out what causes the import error

import numpy as np
class SparseDispatcher(object):

    def __init__(self, num_experts, gates, is_cuda=True):
        """Create a SparseDispatcher."""

        self._gates = gates
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets

        if self.is_cuda:
            self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        else:
            self._part_sizes = list((gates > 0).sum(0).numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

        # # TODO: quick hack
        # self._nonzero_gates = self._nonzero_gates.view(-1, 1, 1)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True).to(self.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined


    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class TokenLevelFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class MoeTokenLevelFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, num_experts=4, k=2, dropout=0.1, is_cuda=True, noisy_gating=True):
        super(MoeTokenLevelFeedForward, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.is_cuda = is_cuda
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList([TokenLevelFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(d_model, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d_model, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]).to(device=self.device), torch.tensor([1.0]).to(device=self.device))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0]).to(self.device) # TODO: requires grad?
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):

        return (gates > 0).sum(0)


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):


        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (torch.arange(batch) * m + self.k).to(self.device)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1).to(self.device)
        is_in = torch.gt(noisy_values, threshold_if_in).to(self.device)
        threshold_positions_if_out = (threshold_positions_if_in - 1).to(self.device)
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1).to(self.device)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train).to(self.device)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev).to(self.device)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True).to(self.device)
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(self.device)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, train=True, loss_coef=0.1):
        gates, load = self.noisy_top_k_gating(x, train)
        # calculate importance loss
        importance = gates.sum(0)
        
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates, is_cuda=self.is_cuda)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss