import numpy as np
import torch

import mha_nested
import mha_padded_autograd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    W_out = torch.randn(E_out, E_total)
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

    # check correctness

    # nested calculation
    out = mha_nested.mha_nested(
        query, key, value, nheads,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v,
        dropout_p=dropout_p)
    out.backward(out.clone())
    # save grads
    W_q_grad = W_q.grad.detach().clone()
    b_q_grad = b_q.grad.detach().clone()
    W_k_grad = W_k.grad.detach().clone()
    b_k_grad = b_k.grad.detach().clone()
    W_v_grad = W_v.grad.detach().clone()
    b_v_grad = b_v.grad.detach().clone()
    W_out_grad = W_out.grad.detach().clone()
    # zero grads
    W_q.grad.detach_().zero_()
    b_q.grad.detach_().zero_()
    W_k.grad.detach_().zero_()
    b_k.grad.detach_().zero_()
    W_v.grad.detach_().zero_()
    b_v.grad.detach_().zero_()
    W_out.grad.detach_().zero_()

    # def mha_nestd_grad_test_func(
    # W_q, W_k, W_v, W_out,
    # b_q, b_k, b_v):
    #     out = mha_nested.mha_nested(
    #         query, key, value, nheads,
    #         W_q, W_k, W_v, W_out,
    #         b_q=b_q, b_k=b_k, b_v=b_v,
    #         dropout_p=dropout_p)
    #     return out.to_padded_tensor(0.0)
    # inputs = (
    #      W_q, W_k, W_v, W_out,
    #      b_q, b_k, b_v)
    # print(torch.autograd.gradcheck(mha_nestd_grad_test_func, inputs=inputs))

    # padded calculation
    padded_out = mha_padded_autograd.mha_padded_autograd(
        padded_query, padded_key, padded_value, nheads, attn_mask,
        W_q, W_k, W_v, W_out,
        b_q=b_q, b_k=b_k, b_v=b_v,
        dropout_p=dropout_p)
    padded_out.backward(padded_out.clone())
    # save grads
    W_q_grad_padded = W_q.grad.detach().clone()
    b_q_grad_padded = b_q.grad.detach().clone()
    W_k_grad_padded = W_k.grad.detach().clone()
    b_k_grad_padded = b_k.grad.detach().clone()
    W_v_grad_padded = W_v.grad.detach().clone()
    b_v_grad_padded = b_v.grad.detach().clone()
    W_out_grad_padded = W_out.grad.detach().clone()

    # compare
    print("forward difference =", (out.to_padded_tensor(0.0, (N, L_t, E_out)) - padded_out).detach_().abs().max())
    print("W_q_grad difference =", (W_q_grad - W_q_grad_padded).abs().max())
    print("b_q_grad difference =", (b_q_grad - b_q_grad_padded).abs().max())
    print("W_k_grad difference =", (W_k_grad - W_k_grad_padded).abs().max())
    print("b_k_grad difference =", (b_k_grad - b_k_grad_padded).abs().max())
    print("W_v_grad difference =", (W_v_grad - W_v_grad_padded).abs().max())
    print("b_v_grad difference =", (b_v_grad - b_v_grad_padded).abs().max())
    print("W_out_grad difference =", (W_out_grad - W_out_grad_padded).abs().max())

    print(W_k_grad, W_k_grad_padded, sep='\n')
    print(b_k_grad, b_k_grad_padded, sep='\n')
