import math
import torch
import torch.nn.functional as F

"""
Args:
    query: query of shape (N, L_t, E_q)
    key: key of shape (N, L_s, E_k)
    value: value of shape (N, L_s, E_v)
    nheads: number of heads in multi-head attention
    attn_mask_q: boolean mask indicating locations that should not take part in attention for query, shape (N, L_t)
    attn_mask_kv: boolean mask indicating locations that should not take part in attention for key and value, shape (N, L_s)
    W_q: Weight for query input projection of shape (E_total, E_q)
    W_k: Weight for key input projection of shape (E_total, E_k)
    W_v: Weight for value input projection of shape (E_total, E_v)
    W_out: Weight for output projection of shape (E_out, E_total)
    b_q (optional): Bias for query input projection of shape E_total. Default: None
    b_k (optional): Bias for key input projection of shape E_total. Default: None
    b_v (optional): Bias for value input projection of shape E_total. Default: None
    b_out (optional): Bias for output projection of shape E_out. Default: None
    dropout_p: dropout probability. Default: 0.0
    where:
        N is the batch size
        L_t is the target sequence length (padded)
        L_s is the source sequence length (padded)
        E_q is the embedding size for query
        E_k is the embedding size for key
        E_v is the embedding size for value
        E_total is the embedding size for all heads combined
        E_out is the output embedding size
Returns:
    attn_output: Output of shape (N, L_t, E_out)
"""
def mha_padded(query, key, value, nheads,
attn_mask_q, attn_mask_kv,
W_q, W_k, W_v, W_out,
b_q=None, b_k=None, b_v=None, b_out=None,
dropout_p=0.0):
    N = query.size(0)
    L_t = query.size(1)
    L_s = key.size(1)
    E_total = W_q.size(0)
    assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
    E_head = E_total // nheads

    # apply input projection
    # (N, L_t, E_q) -> (N, L_t, E_total)
    query = F.linear(query, W_q, b_q)
    # (N, L_s, E_k) -> (N, L_s, E_total)
    key = F.linear(key, W_k, b_k)
    # (N, L_s, E_v) -> (N, L_s, E_total)
    value = F.linear(value, W_v, b_v)

    # padding-specific step: remove bias from padded entries
    # in the specific multihead-attention formula it is not necessary to remove these bias
    # because the -inf padding later on in softmax step can take care of it
    # but to be general here we demonstrate the bias removal
    for i in range(N):
        for j in range(L_t):
            if attn_mask_q[i, j]:
                query[i, j, :] = 0.0
        for j in range(L_s):
            if attn_mask_kv[i, j]:
                key[i, j, :] = 0.0
                value[i, j, :] = 0.0

    # reshape query, key, value to separate by head
    # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head) -> (N * nheads, L_t, E_head)
    query = query.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head) -> (N * nheads, L_s, E_head)
    key = key.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head) -> (N * nheads, L_s, E_head)
    value = value.reshape(N, -1, nheads, E_head).transpose(1, 2).reshape(N * nheads, -1, E_head)

    # query bmm key^T
    # (N * nheads, L_t, E_head) x (N * nheads, L_s, E_head)^T -> (N * nheads, L_t, L_s)
    keyT = key.transpose(-1, -2)
    # padding-specific step: add -inf mask for padding in softmax
    attn_mask = query.new_zeros((N, nheads, L_t, L_s))
    for i in range(N):
        for j in range(L_t):
            for k in range(L_s):
                if attn_mask_q[i, j] or attn_mask_kv[i, k]:
                    attn_mask[i, :, j, k] = float("-inf")
    attn_mask = attn_mask.reshape((N * nheads, L_t, L_s))
    attn_weights = torch.baddbmm(attn_mask, query, keyT)
    # if no padding, it could have been as simple as
    #     attn_weights = torch.bmm(query, keyT)

    # scale down
    attn_weights = attn_weights * (1.0 / math.sqrt(E_head))

    # softmax
    attn_weights = F.softmax(attn_weights, dim=-1).nan_to_num_(0.0)

    # dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # attention_weights bmm value
    # (N * nheads, L_t, L_s) x (N * nheads, L_s, E_head) -> (N * nheads, L_t, E_head)
    attn_output = attn_weights.bmm(value)

    # merge heads
    # (N * nheads, L_t, E_head) -> (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
    attn_output = attn_output.reshape(N, nheads, -1, E_head).transpose(1, 2).reshape(N, -1, E_total)

    # apply output projection
    # (N, L_t, E_total) -> (N, L_t, E_out)
    attn_output = F.linear(attn_output, W_out, b_out)

    # padding-specific step: remove output projection bias from padded entries
    for i in range(N):
        for j in range(L_t):
            if attn_mask_q[i, j]:
                attn_output[i, j, :] = 0.0

    return attn_output
