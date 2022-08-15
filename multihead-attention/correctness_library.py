import numpy as np
import torch

import mha_nested
import mha_padded

if __name__ == "__main__":
    N, L = 2, 4
    E = 6
    nheads = 3
    dropout_p = 0.0

    rng = np.random.default_rng()

    # create parameters
    mha_lib = torch.nn.MultiheadAttention(E, nheads, batch_first=True)
    mha_lib.eval()
    mha_lib.in_proj_weight.requires_grad_(False)
    mha_lib.in_proj_bias.requires_grad_(False)
    mha_lib.out_proj.weight.requires_grad_(False)
    mha_lib.out_proj.bias.requires_grad_(False)
    W_q, b_q = mha_lib.in_proj_weight[: E, :], mha_lib.in_proj_bias[: E]
    W_k, b_k = mha_lib.in_proj_weight[E : 2 * E, :], mha_lib.in_proj_bias[E : 2 * E]
    W_v, b_v = mha_lib.in_proj_weight[2 * E :, :], mha_lib.in_proj_bias[2 * E :]
    W_out, b_out = mha_lib.out_proj.weight, mha_lib.out_proj.bias
    # nested input
    queries = []
    keys    = []
    values  = []
    for i in range(N):
        l = rng.integers(1, L + 1)
        queries.append(torch.randn((l, E)))
        keys   .append(torch.randn((l, E)))
        values .append(torch.randn((l, E)))
    query = torch.nested_tensor(queries)
    key   = torch.nested_tensor(keys   )
    value = torch.nested_tensor(values )
    # pad input
    padded_query = query.to_padded_tensor(0.0, (N, L, E))
    padded_key   = key  .to_padded_tensor(0.0, (N, L, E))
    padded_value = value.to_padded_tensor(0.0, (N, L, E))
    attn_mask_q = torch.zeros((N, L), dtype=torch.bool)
    attn_mask_kv = torch.zeros((N, L), dtype=torch.bool)
    for i in range(N):
        for j in range(L):
            if padded_query[i, j, 0].item() == 0.0:
                attn_mask_q[i, j] = True
                attn_mask_kv[i, j] = True

    # check correctness
    # nested calculation
    out = mha_nested.mha(
        query, query, query, nheads,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
        dropout_p=dropout_p)
    # padded calculation
    padded_out = mha_padded.mha(
        padded_query, padded_query, padded_query, nheads,
        attn_mask_q, attn_mask_q,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
        dropout_p=dropout_p)
    # library calculation
    out_lib, out_lib_weights = mha_lib(query, query, query)
    # compare
    print((out.to_padded_tensor(0.0, (N, L, E)) - padded_out).abs().max().item())
    print((out.to_padded_tensor(0.0) - out_lib.to_padded_tensor(0.0)).abs().max().item())
