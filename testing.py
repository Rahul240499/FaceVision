import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1

import torch

# List available devices
print(f"Available CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
