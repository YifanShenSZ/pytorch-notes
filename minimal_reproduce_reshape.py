import torch

@torch.inference_mode()
def unary(x):
    y = x.reshape(2, -1, 2, 3)
    return y

# with torch.inference_mode():
if __name__ == "__main__":
    x = torch.nested_tensor([torch.randn((2, 6)), torch.randn((3, 6))])
    y = unary(x)
    print(y)
