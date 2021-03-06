import torch
import torch.nn as nn
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
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets

        if self.is_cuda:
            self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        else:
            self._part_sizes = list((gates > 0).sum(0).numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

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


class MoeTokenLevelFeedForwardGshard(nn.Module):

    def __init__(self, d_model, d_ff, num_experts=4, k=2, dropout=0.1, is_cuda=True, noisy_gating=True):
        super(MoeTokenLevelFeedForwardGshard, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.is_cuda = is_cuda
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList(
            [TokenLevelFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout) for _ in range(self.num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(d_model, num_experts), requires_grad=True)

        assert (self.k <= self.num_experts)


    def _gates_to_load(self, gates):

        return (gates > 0).sum(0)

    def expert_1_and_loss(self, x):
        group_combine_weights_1 = torch.zeros(size=(x.size(0), self.num_experts), requires_grad=True).to(device=self.device)
        gating_decision_per_expert = torch.zeros(size=(self.num_experts,), requires_grad=True).to(device=self.device)
        gates_per_token_per_expert = F.softmax(x @ self.w_gate, dim=1)
        mean_gates_per_expert = torch.mean(gates_per_token_per_expert, dim=0)

        top_gates, top_indices = gates_per_token_per_expert.topk(k=min(self.k, self.num_experts), dim=1)
        gates_sum = torch.sum(top_gates, dim=1)
        top_gates = torch.div(top_gates, gates_sum.unsqueeze(1))

        expert_decisions_count = torch.empty_like(gating_decision_per_expert, requires_grad=False).to(device=self.device)

        for i in range(self.num_experts):
            # produce the if mask in here
            expert_decisions_count[i] = torch.eq(top_indices[:, 0], i).sum().item()

        loss_aux = torch.mean(torch.div(expert_decisions_count, x.size(0)) * mean_gates_per_expert)


        group_combine_weights_1 = group_combine_weights_1.scatter(1, top_indices[:, 0].unsqueeze(dim=1),
                                                                  top_gates[:, 0].unsqueeze(dim=1))

        return group_combine_weights_1, loss_aux, top_gates, top_indices


    def expert_2(self, x, top_gates, top_indices):
        # second expert
        group_combine_weights_2 = torch.zeros(size=(x.size(0), self.num_experts), requires_grad=True).to(device=self.device)

        random_uniform = torch.div(torch.rand_like(group_combine_weights_2), 2)

        group_combine_weights_2 = group_combine_weights_2.scatter(1, top_indices[:, 1].unsqueeze(dim=1),
                                                                  top_gates[:, 1].unsqueeze(dim=1))

        group_combine_weights_2 = torch.where(group_combine_weights_2 > random_uniform, group_combine_weights_2,
                                              torch.zeros_like(group_combine_weights_2))

        return group_combine_weights_2


    def random_routing_for_unutilised_tokens(self, group_combine_weights_with_capacity):

        unutilised_token_indicies = (((group_combine_weights_with_capacity > 0).sum(1) == 0).nonzero()).squeeze(dim=1)

        if unutilised_token_indicies.size(0) > 0:
            random_values = torch.rand(size=(unutilised_token_indicies.size(0), self.k), requires_grad=True).to(
                device=self.device)
            random_values_sum = torch.sum(random_values, dim=1)
            random_values = torch.div(random_values, random_values_sum.unsqueeze(1))

            random_indicies = torch.randint_like(input=random_values, low=0, high=self.num_experts, dtype=torch.int64).to(
                device=self.device)

            placeholder_tensor = torch.zeros_like(group_combine_weights_with_capacity[unutilised_token_indicies, :],
                                                  requires_grad=True).to(device=self.device)

            placeholder_tensor = placeholder_tensor.scatter(1, random_indicies, random_values)

            group_combine_weights_with_capacity[unutilised_token_indicies, :] = placeholder_tensor

        return group_combine_weights_with_capacity



    def generate_gates_and_loss(self, x):

        group_combine_weights_with_capacity = torch.zeros(size=(x.size(0), self.num_experts), requires_grad=True).to(
            device=self.device)
        expert_capacity = int(x.size(0) / self.num_experts)

        group_combine_weights_1, loss_aux, top_gates, top_indices = self.expert_1_and_loss(x=x)

        group_combine_weights_2 = self.expert_2(x=x, top_gates=top_gates, top_indices=top_indices)

        group_combine_weights = group_combine_weights_1 + group_combine_weights_2

        top_values, top_new_indicies = group_combine_weights.topk(k=expert_capacity, dim=0)

        group_combine_weights_with_capacity = group_combine_weights_with_capacity.scatter(0, top_new_indicies,
                                                                                          top_values)

        group_combine_weights_with_capacity = self.random_routing_for_unutilised_tokens(group_combine_weights_with_capacity=group_combine_weights_with_capacity)

        return group_combine_weights_with_capacity, loss_aux

    def forward(self, x, train=True, loss_coef = 0.1, performLogging=False):
        gates, loss = self.generate_gates_and_loss(x)

        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates, is_cuda=self.is_cuda)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss