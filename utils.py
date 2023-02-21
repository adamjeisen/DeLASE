import numpy as np
import torch

def numpy_torch_conversion(x, use_torch, device=None):
    if use_torch:
        if isinstance(x, np.ndarray):
            if device is None:
                raise ValueError("If use_torch is True and x is np.ndarray the device must be specified!")
            x = torch.from_numpy(x).to(device)
        else:
            if device is not None:
                if isinstance(device, torch.device):
                    if x.device.index != device.index:
                        x = x.to(device)
                elif isinstance(device, int):
                    if x.device.index != device:
                        x = x.to(device)
                else: # isinstance(device, str)
                    if x.device.type != device:
                        x = x.to(device)    
    else:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
    
    return x