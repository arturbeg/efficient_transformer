import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from multiheaded_attention import MultiheadAttention
import numpy as np
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, experts, query, key, value, key_padding_mask, need_weights, attn_mask):
        # query_exp = query[self._batch_index].squeeze(1)
        # key_exp = key[self._batch_index].squeeze(1)
        # value_exp = value[self._batch_index].squeeze(1)

        query_exp = query[:, self._batch_index, :]
        key_exp = key[:, self._batch_index, :]
        value_exp = value[:, self._batch_index, :]


        # Dimension changed to 1 since the batch is along the first dimension and not the first one like in the original paper
        queries_exp = torch.split(query_exp, self._part_sizes, dim=1)
        keys_exp = torch.split(key_exp, self._part_sizes, dim=1)
        values_exp = torch.split(value_exp, self._part_sizes, dim=1)



        expert_outputs = []
        # TODO: handle cases where one of the experts gets assigned to observations, could populate with zeros?

        for i in range(len(experts)):

            # TODO: refactor, make sure actually requires_grad
            if queries_exp[i].size(1) == 0:
                print("One of the experts has no observations")
                # no observations were passed onto this expert
                attn_zeros_out = torch.zeros(queries_exp[i].size(0), 0, queries_exp[i].size(2), requires_grad=True)

                attn_zeros_weights = torch.zeros(0, queries_exp[i].size(0),
                                            queries_exp[i].size(0), requires_grad=True)
                expert_outputs.append((attn_zeros_out, attn_zeros_weights))

            else:
                expert_outputs.append(experts[i](queries_exp[i], keys_exp[i], values_exp[i], key_padding_mask, need_weights, attn_mask))

        return expert_outputs


    def combine(self, expert_out, multiply_by_gates=True):
        # expert_out_
        expert_out_attn_output = [expert_out[i][0] for i in range(len(expert_out))]
        expert_out_attn_output_weights = [expert_out[i][1] for i in range(len(expert_out))]

        # TODO: why exponentiate and then go back to the log space?
        # apply exp to expert outputs, so we are not longer in log space
        stitched_out = torch.cat(expert_out_attn_output, 1).exp() #attn_output: :math:`(L, N, E)`
        stitched_weights = torch.cat(expert_out_attn_output_weights, 0).exp() #  attn_output_weights: :math:`(N, L, S)` where N is the batch size,


        # TODO: fix multiplication by gates (fix later)
        # if multiply_by_gates:
        #     stitched_out = stitched_out.mul(self._nonzero_gates)
        #     stitched_weights = stitched_weights.mul(self._nonzero_gates)

        zeros_out = torch.zeros(expert_out_attn_output[-1].size(0), self._gates.size(0), expert_out_attn_output[-1].size(2), requires_grad=True)
        zeros_weights = torch.zeros(self._gates.size(0), expert_out_attn_output_weights[-1].size(1), expert_out_attn_output_weights[-1].size(2), requires_grad=True)

        # combine samples that have been processed by the same k experts
        combined_out = zeros_out.index_add(1, self._batch_index, stitched_out.float())
        combined_weights = zeros_weights.index_add(0, self._batch_index, stitched_weights.float())


        # add eps to all zero values in order to avoid nans when going back to log space
        combined_out[combined_out == 0] = np.finfo(float).eps
        combined_weights[combined_weights == 0] = np.finfo(float).eps

        # back to log space
        return combined_out.log(), combined_weights.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




class MoE(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, num_experts, noisy_gating=True, k=2):
        # TODO: k is 2
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        # self.output_size = output_size
        # self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList(MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout) for i in range(self.num_experts))
        self.w_gate = nn.Parameter(torch.zeros(embed_dim, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(embed_dim, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)


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
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out), 1)
        # is each value currently in the top k.
        prob_if_in = self.normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = self.normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        '''
        Shape of x is [10, 32, 512] where; x is the query;
        TODO: could convert the sequence into bag of words instead ?
        TODO: can also pass each word through the w_gate and them sum the resulted logits?? --> the current approach
        TODO: sum the resulted logits using some sort of latent layer (similar to the one that will be used for the MoG
        TODO: change the shape of the query, figure out the best way to reshape the query
        TODO: make sure the shape inconsistencies are handeled since original moe paper assumes a shape of
        '''

        # TODO: turn into a separate method?
        clean_logits = torch.tensor((), dtype=torch.float)
        clean_logits = clean_logits.new_zeros((x.size(1), self.num_experts)) # x.size(1) refers to the batch size


        for i in range(len(x)):
            # TODO: perhaps apply another weight matrix in this sum
            clean_logits += x[i] @ self.w_gate

        if self.noisy_gating:
            # raw_noise_stddev = x @ self.w_noise
            raw_noise_stddev = torch.tensor((), dtype=torch.float)
            raw_noise_stddev = raw_noise_stddev.new_zeros((x.size(1), self.num_experts))

            for i in range(len(x)):
                raw_noise_stddev += x[i] @ self.w_noise

            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load



    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, train=True, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(query, train)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_outputs = dispatcher.dispatch(self.experts, query, key, value, key_padding_mask, need_weights, attn_mask)
        gates = dispatcher.expert_to_gates()
        attn_output, attn_output_weights = dispatcher.combine(expert_outputs)
        return attn_output, loss, attn_output_weights