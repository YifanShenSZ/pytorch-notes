# Dispatcher
PyTorch wiki [core frontend onboarding](https://github.com/pytorch/pytorch/wiki/Core-Frontend-Onboarding) unit "dispatcher, structured kernels, and codegen" is a great resource for dispatcher. Here are a few gotcha in addition to that.

## codegen
The generated kernels reside in "build/aten/src/ATen/Register`dispatch key name`.cpp". In another word, they are registered at that c++ file.

## priority
The general priority is:
```
Autograd* > * > Composite*
```

For example:
```
AutogradNestedTensor > NestedTensorCPU > CompositeImplicitAutograd
```

`Autograd*` is the kernel to run with autograd engine and has the highest priority, so it would almost always get called, unless you use something like `torch.inference_mode` to disable `Autograd*`

`Composite*` is a wild card for a set of keys, which means a bunch of keys will have their kernel registered to the composite kernel, unless a specific kernel is provided for a key

## overload
Take `select` as an example, in "aten/src/ATen/native/native_funcrtions.yaml":
```
- func: select.Dimname(Tensor(a) self, Dimname dim, int index) -> Tensor(a)
  variants: function, method
  device_check: NoCheck
  device_guard: False

- func: select.int(Tensor(a) self, int dim, int index) -> Tensor(a)
  variants: function, method
  device_check: NoCheck
  device_guard: False
  dispatch:
    CompositeExplicitAutograd: select
    SparseCsrCPU, SparseCsrCUDA: select_sparse_csr
    NestedTensorCPU, NestedTensorCUDA: select_nested
```
These are both `select` operations, where ".Dimname" and ".int" means they are overloads: in c++ overloads have same name but different arguments, in "native_functions.yaml" we indicate that by ".\*"

## code
There are 3 major sets of codes:
1. "c10/core/DispatchKey.\*" define dispatch keys and "c10/core/DispatchKeySet.\*" define dispatch key sets
2. codegen generates and registers functions actually called
3. "aten/src/ATen/core/dispatch/OperatorEntry.\*" calculate dispatch table then "aten/src/ATen/core/dispatch/Dispatcher.\*" dispatch

## debug tools
`torch._C._dispatch_dump("aten::operation")` dumps the raw dispatch information registered to an aten::operation

`torch._C._dispatch_dump_table("aten::op")` dumps the dispatch table computed from raw dispatch information for an aten::operation
