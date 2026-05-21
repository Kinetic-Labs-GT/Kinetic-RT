import torch

class KernelTuner:
    def __init__(self):
        pass

    def get_warp_size(self, device_id=0):
        return torch.cuda.get_device_properties(device_id).warp_size
