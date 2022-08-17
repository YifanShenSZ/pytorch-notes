# Autograd
PyTorch wiki [core frontend onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) unit "Autograd" and doc [autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html?highlight=autograd) are great resources for autograd. Here are a few gotcha in addition to that.

## view functions
1st, view functions need special care since their output share memory with input, which means their in-place operations must be tracked carefully

2nd, view function mutates only tensor metadata, so in backward it can be treated:
1. with special optimization (fast path)
2. as a general function (slow path): let autograd engine record every view operation

For tensors who do not have regular metadata (e.g. nested tensor has different sizes and strides), we have to use slow path for now
