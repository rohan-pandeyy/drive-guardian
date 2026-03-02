# PyTorch with CUDA Installation Guide (Windows)

To unlock the performance of the Drive Guardian ML models (YOLO / ONNX), you must install PyTorch with GPU acceleration.

By default, the `requirements.txt` file will install `torch` and `torchvision` using standard Pip syntax, which usually results in a CPU-only binary being downloaded.

To override this and utilize NVIDIA GPUs with CUDA support, follow these steps:

## Prerequisites

1.  An NVIDIA GPU with modern drivers.
2.  Your Drive Guardian Python Virtual Environment (`.venv`) is activated.

## Installation

Because CUDA and PyTorch versions are tightly coupled to your computer's specific NVIDIA drivers, you **must find the correct installation command for your setup**.

1. Open your terminal or command prompt and run:

    ```bash
    nvidia-smi
    ```

    Look at the top right of the output table to find your **CUDA Version** limit (e.g., `CUDA Version: 13.1`).

2. Go to the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
3. In the setup matrix, select your preferences:
    - **OS**: Windows
    - **Package**: Pip
    - **Compute Platform**: Select a CUDA version that is _less than or equal to_ the version you saw in `nvidia-smi` (e.g., CUDA 12.1, 12.4, etc).
4. Copy the generated `pip install` command and run it in your `.venv`. It will look something like this:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verification

After installation concludes, run the provided diagnostic script from the root of the repository to confirm PyTorch can communicate with your GPU:

```bash
python diagnostics/test_gpu.py
```

**Expected Output:**

```text
--- GPU Diagnostic ---
CUDA Available (PyTorch): True
CUDA Device Name: NVIDIA GeForce RTX [Your GPU Model]
YOLO Ultralytics Device: cpu
...
```

_Note: It is expected that YOLO initially initializes on the CPU in the diagnostic script unless explicitly commanded otherwise in the code implementation._
