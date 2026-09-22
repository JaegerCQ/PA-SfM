# Online installation only

This directory contains no wheels. Install the exact PA-SfM dependencies in `../requirements.pip-hash-lock.txt` online, after creating the Conda environment from `../conda-explicit.txt`. SHA256 values come from published PyPI distribution metadata and the official PyTorch CUDA 12.1 index; `../pip-hash-provenance.json` records every source and archive. No wheel archives were downloaded or bundled.

The original repository's CUDA 12.6 wheelhouse consisted of Git LFS pointers and is not included. The unrelated, unused `opencv-python-headless==5.0.0.93` is excluded from both installation files because it requires NumPy >=2, whereas this PA-SfM environment uses NumPy 1.26.4. `../installed-pip-freeze.txt` preserves the complete original installed-package snapshot, including OpenCV.
