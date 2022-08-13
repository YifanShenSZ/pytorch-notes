'''
Dump the dispatch table computed from raw dispatch information for an aten::operation
'''

import torch

print(torch._C._dispatch_dump_table("aten::reshape"))
