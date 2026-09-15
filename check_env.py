"""Check installed packages and run a small CUDA operation; downloads no model."""
from importlib.metadata import version

import torch
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

for package in ("lerobot", "torch", "torchvision", "transformers", "torchcodec"):
    print(f"{package}: {version(package)}")
print(f"CUDA runtime: {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA is unavailable"
print(f"GPU: {torch.cuda.get_device_name(0)}")
x = torch.randn(32, 32, device="cuda")
assert torch.isfinite(x @ x.T).all().item()
torch.cuda.synchronize()
config = SmolVLAConfig()
print(f"SmolVLA import OK; default chunk_size={config.chunk_size}")
print("CUDA check passed. Model/dataset training has not been run.")
