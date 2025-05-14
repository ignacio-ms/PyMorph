import sys
import torch

print(f"Python version: {sys.version}")
print(f"Python ext: {sys.executable}")
print(f"GPU activated: {torch.cuda.is_available()}")