"""
Common utilities and setup for TASTE-SpokenLM streaming component tests.
Supports CPU-only testing via NO_CUDA environment variable.
"""

import torch
import os
import sys

# Add the parent directory to path to import taste_speech
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NO_CUDA toggle for CPU-only testing (e.g., on Mac without GPU)
NO_CUDA = os.environ.get('NO_CUDA', 'True').lower() in ('true', '1', 'yes', 'on')

# Device configuration based on NO_CUDA flag
if NO_CUDA:
    DEVICE = torch.device('cpu')
    print("Running tests in CPU-only mode (NO_CUDA=True)")
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running tests with device: {DEVICE}")

# Force CPU-only mode if CUDA is not available regardless of NO_CUDA setting
if not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
    NO_CUDA = True