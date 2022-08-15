import numpy as np
import torch

import mha_nested
import mha_padded

if __name__ == "__main__":
    N, L_t, L_s = 2, 4, 4
    E_q, E_k, E_v, E_total, E_out = 6, 6, 6, 6, 6
    nheads = 3
    dropout_p = 0.0

    rng = np.random.default_rng()

    # create parameters
    W_q, b_q = torch.randn(E_total, E_q), torch.randn(E_total)
    W_k, b_k = torch.randn(E_total, E_k), torch.randn(E_total)
    W_v, b_v = torch.randn(E_total, E_v), torch.randn(E_total)
    W_out, b_out = torch.randn(E_out, E_total), torch.randn(E_out)
    # nested input
    queries = []
    keys    = []
    values  = []
    for i in range(N):
        l_t = rng.integers(1, L_t + 1)
        l_s = rng.integers(1, L_s + 1)
        queries.append(torch.randn((l_t, E_q)))
        keys   .append(torch.randn((l_s, E_k)))
        values .append(torch.randn((l_s, E_v)))
    query = torch.nested_tensor(queries)
    key   = torch.nested_tensor(keys   )
    value = torch.nested_tensor(values )
    # pad input
    padded_query = query.to_padded_tensor(0.0, (N, L_t, E_q))
    padded_key   = key  .to_padded_tensor(0.0, (N, L_s, E_k))
    padded_value = value.to_padded_tensor(0.0, (N, L_s, E_v))
    attn_mask_q = torch.zeros((N, L_t), dtype=torch.bool)
    attn_mask_kv = torch.zeros((N, L_s), dtype=torch.bool)
    for i in range(N):
        for j in range(L_t):
            if padded_query[i, j, 0].item() == 0.0:
                attn_mask_q[i, j] = True
        for j in range(L_s):
            if padded_key[i, j, 0].item() == 0.0:
                attn_mask_kv[i, j] = True

    # check correctness
    # nested calculation
    out = mha_nested.mha_nested(
        query, key, value, nheads,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
        dropout_p=dropout_p)
    # padded calculation
    padded_out = mha_padded.mha_padded(
        padded_query, padded_key, padded_value, nheads,
        attn_mask_q, attn_mask_kv,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v, b_out=b_out,
        dropout_p=dropout_p)
    # compare
    print(out, padded_out, sep='\n')
    print((out.to_padded_tensor(0.0, (N, L_t, E_out)) - padded_out).abs().max().item())
