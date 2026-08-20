"""Agent package configuration."""

import os
import platform

from dotenv import load_dotenv

load_dotenv()

# Fixes an XGBoost + PyTorch OpenMP crash that only happens on macOS.
if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"