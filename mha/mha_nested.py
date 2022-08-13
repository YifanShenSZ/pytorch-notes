import math
import torch
import torch.nn.functional as F

"""
Args:
    query: query of shape (N, L_t, E_q)
    key: key of shape (N, L_s, E_k)
    value: value of shape (N, L_s, E_v)
    nheads: number of heads in multi-head attention
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
        L_t is the target sequence length (jagged)
        L_s is the source sequence length (jagged)
        E_q is the embedding size for query
        E_k is the embedding size for key
        E_v is the embedding size for value
        E_total is the embedding size for all heads combined
        E_out is the output embedding size
Returns:
    attn_output: Output of shape (N, L_t, E_out)
"""
def mha_nested(query, key, value, nheads,
W_q, W_k, W_v, W_out,
b_q=None, b_k=None, b_v=None, b_out=None,
dropout_p=0.0):
    N = query.size(0)
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

    # reshape query, key, value to separate by head
    # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
    query = query.reshape(-1, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    key = key.reshape(-1, -1, nheads, E_head).transpose(1, 2)
    # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
    value = value.reshape(-1, -1, nheads, E_head).transpose(1, 2)

    # query matmul key^T
    # (N, nheads, L_t, E_head) x (N, nheads, L_s, E_head)^T -> (N, nheads, L_t, L_s)
    keyT = key.transpose(-1, -2)
    attn_weights = torch.matmul(query, keyT)

    # scale down
    attn_weights = attn_weights * (1.0 / math.sqrt(E_head))

    # softmax
    attn_weights = F.softmax(attn_weights, dim=-1)

    # dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # attention_weights matmul value
    # (N, nheads, L_t, L_s) x (N, nheads, L_s, E_head) -> (N, nheads, L_t, E_head)
    attn_output = torch.matmul(attn_weights, value)

    # merge heads
    # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
    attn_output = attn_output.transpose(1, 2).reshape(N, -1, E_total)

    # apply output projection
    # (N, L_t, E_total) -> (N, L_t, E_out)
    attn_output = F.linear(attn_output, W_out, b_out)

    return attn_output
