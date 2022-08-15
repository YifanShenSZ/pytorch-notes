# Multi-head Attention
Multi-head attention is the core component of transformer (fun fact: the original transformer paper is [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf))

"mha_nested.py" implements general multi-head attention with nested tensor

"mha_padded_fast.py" and "mha_padded_autograd.py" implements a special version with padded regular tensor: the bias in output projection has to be 0. The general version in "mha_padded.py" is super slow, mainly due to the bias removal after linear operation

## benchmark
We do not need to use the whole real sentences; the lengths of sentences are enough (then we embed with random numbers)
* wikitext-2: sentence lengths from wikitext-2 corpus
* zipf: from wikitext-2 corpus we know the ranks of end-of-sentence punctuations, then we generate sentence lengths with user-defined alpha value

correctness:
* nested version vs. padded versions
* In addition, in `torch.nn` there is `torch.nn.MultiheadAttention`, which can use nested tensor in a very specific case (self-attention), so we also compare to it

performance:
* inference: nested version vs. padded fast version
* training: nested version vs. padded autograd version
