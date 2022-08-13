import argparse
import numpy as np
import torch
import timeit

import mha_nested
import mha_padded_autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def noncontiguous_to_padded_tensor(input, shape=None):
    if shape is None:
        # for now to_padded_tensor gives the correct sizes
        # but wrong entries when the buffer memory is discontinuous
        continuous_to_padded_tensor = input.to_padded_tensor(0.0)
        shape = continuous_to_padded_tensor.shape
    tensors = input.unbind()
    assert len(tensors) > 0
    result = tensors[0].new_zeros(shape)
    for itensor in range(len(tensors)):
        tensor = tensors[itensor]
        view = result[itensor]
        for idim in range(tensor.dim()):
            view = view.narrow(idim, 0, tensor.size(idim))
        view.copy_(tensor)
    return result

if __name__ == "__main__":
    N = 512
    E_q, E_k, E_v, E_total, E_out = 512, 512, 512, 512, 512
    nheads = 8

    dropout_p = 0.3

    sequence_lengths = []
    with open("sequence_lengths.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            sequence_lengths.append(int(line))
    sequence_lengths = np.random.choice(np.array(sequence_lengths), size=(N))
    L_t = np.max(sequence_lengths)
    L_s = L_t

    # create parameters.
    W_q, b_q = torch.randn((E_total, E_q), device=device), torch.randn(E_total, device=device)
    W_k, b_k = torch.randn((E_total, E_k), device=device), torch.randn(E_total, device=device)
    W_v, b_v = torch.randn((E_total, E_v), device=device), torch.randn(E_total, device=device)
    W_out = torch.randn((E_out, E_total), device=device)
    # create nested input
    queries = []
    keys    = []
    values  = []
    for i in range(N):
        l_t = sequence_lengths[i]
        l_s = l_t
        queries.append(torch.randn((l_t, E_q), device=device))
        keys   .append(torch.randn((l_s, E_k), device=device))
        values .append(torch.randn((l_s, E_v), device=device))
    query = torch.nested_tensor(queries)
    key   = torch.nested_tensor(keys   )
    value = torch.nested_tensor(values )
    # pad input
    padded_query = query.to_padded_tensor(0.0, (N, L_t, E_q))
    padded_key   = key  .to_padded_tensor(0.0, (N, L_s, E_k))
    padded_value = value.to_padded_tensor(0.0, (N, L_s, E_v))
    attn_mask = padded_query.new_zeros((N, nheads, L_t, L_s), device=device)
    for batch in range(N):
        for i in range(L_t):
            for j in range(L_s):
                if padded_query[batch, i, 0].item() == 0.0 or padded_key[batch, j, 0].item() == 0.0:
                    attn_mask[batch, :, i, j].fill_(1.0)

    # turn on gradient
    W_q.requires_grad_(True)
    b_q.requires_grad_(True)
    W_k.requires_grad_(True)
    b_k.requires_grad_(True)
    W_v.requires_grad_(True)
    b_v.requires_grad_(True)
    W_out.requires_grad_(True)

    # check performance
    t0 = timeit.default_timer()
    out = mha_nested.mha_nested(
        query, key, value, nheads,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v,
        dropout_p=dropout_p)
    out.backward(out.clone())
    t1 = timeit.default_timer()
    out_padded = mha_padded_autograd.mha_padded_autograd(
        padded_query, padded_key, padded_value, nheads, attn_mask,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v,
        dropout_p=dropout_p)
    out_padded.backward(out_padded.clone())
    t2 = timeit.default_timer()
    # output
    print(np.sum(L_t - sequence_lengths) / (N * L_t), t1 - t0, t2 - t1, sep='\t')
