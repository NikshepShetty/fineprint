"""Agent package configuration."""

import os
import platform

# Fixes an XGBoost + PyTorch OpenMP crash that only happens on macOS.
# Must run before either library is imported.
if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
