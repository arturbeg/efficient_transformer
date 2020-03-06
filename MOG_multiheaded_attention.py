import torch
import torch.nn as nn
from torch.distributions.normal import Normal
# from mlp import MLP
from multiheaded_attention import MultiheadAttention
import numpy as np
# TODO: rewrite the docstrings
# TODO: make more general purporse
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
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # query_exp = query[self._batch_index].squeeze(1)
        # key_exp = key[self._batch_index].squeeze(1)
        # value_exp = value[self._batch_index].squeeze(1)

        query_exp = query[:, self._batch_index, :]
        key_exp = key[:, self._batch_index, :]
        value_exp = value[:, self._batch_index, :]


        # Dimension changed to 1 since the batch is along the first dimension and not the first one like in the original
        # paper
        queries_exp = torch.split(query_exp, self._part_sizes, dim=1)
        keys_exp = torch.split(key_exp, self._part_sizes, dim=1)
        values_exp = torch.split(value_exp, self._part_sizes, dim=1)



        expert_outputs = []

        for i in range(len(experts)):
            expert_outputs.append(experts[i](queries_exp[i], keys_exp[i], values_exp[i], key_padding_mask, need_weights, attn_mask))

        return expert_outputs


    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """




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
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)




# TODO: refactor / OOP principles
class MoG(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, embed_dim, num_heads, dropout, num_experts, noisy_gating=True, k=2):
        # TODO: k is 2
        super(MoG, self).__init__()
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
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        return x.float().var() / (x.float().mean()**2 + eps)


    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)




    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """

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


    def mog_gating(self, x):

        # Convert x into a latent space ?
        # run through each Gaussian
        # find the top k Gaussians associated with the top experts for a given x
        # proceed as before



        pass


    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        # TODO: change the shape of the query, figure out the best way to reshape the query
        # TODO: make sure the shape inconsistencies are handeled since original moe paper assumes a shape of
        # [batch_size, input_size]

        # x is the query

        '''
        Shape of x is [10, 32, 512] where 
        '''

        # TODO: could convert the sequence into bag of words instead ?
        # TODO: can also pass each word through the w_gate and them sum the resulted logits?? --> the current appraoch




        x = x[0]


        clean_logits = x @ self.w_gate


        if self.noisy_gating:
            raw_noise_stddev = x @ self.w_noise
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
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(query, train)
        # what do we use to determine the gating (key, value or query)?
        # the query seems more intuitive
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_outputs = dispatcher.dispatch(self.experts, query, key, value, key_padding_mask, need_weights, attn_mask)
        gates = dispatcher.expert_to_gates()
        attn_output, attn_output_weights = dispatcher.combine(expert_outputs)
        return attn_output, loss, attn_output_weights
