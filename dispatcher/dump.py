'''
Dump the raw dispatch information registered to an aten::operation
'''

import torch

print(torch._C._dispatch_dump("aten::reshape"))
