import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    tl.store(y_ptr + offsets, x + 1)

x = torch.zeros(16, device='cpu')
y = torch.zeros(16, device='cpu')
try:
    add_kernel[(1,)](x, y, BLOCK_SIZE=16)
    print("Success")
except Exception as e:
    print("Error:", e)
