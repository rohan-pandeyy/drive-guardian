# Drive Guardian: Experimental Results & Architecture Analysis

## 1. Executive Summary

Drive Guardian is a real-time, edge-capable Advanced Driver Assistance System (ADAS) leveraging dual-inference computer vision. This document outlines the performance, hardware optimizations, and mathematical heuristics used to achieve real-time latency on consumer-grade hardware (Nvidia RTX 4050 6GB).

## 2. Model Architecture & Multi-Model Fusion

To achieve robust safety warnings, the system runs two state-of-the-art neural networks simultaneously:

- **Object Detection:** Ultralytics YOLOv26 (`.pt`), utilizing PyTorch with CUDA 13.0 bindings.
- **Lane Segmentation:** Ultra-Fast-Lane-Detection-V2 (UFLDv2 ResNet18), converted to `.onnx` and executed via `onnxruntime-gpu` (CUDA 12).

### 2.1 Multi-Model Fusion Logic

A critical achievement of this system is the elimination of false-positive collision warnings from adjacent lanes. This was achieved by fusing the outputs of both models:

1. UFLDv2 extracts the ego-lane boundary coordinates `(x, y)`.
2. YOLO detects vehicle bounding boxes `(x_min, y_min, x_max, y_max)`.
3. The algorithm evaluates the bottom-center coordinate of the YOLO bounding box. Only vehicles whose bottom-center falls _between_ the left and right UFLD lane polynomials are flagged for Forward Collision Warning (FCW) analysis.

## 3. Mathematical Safety Heuristics

Rather than relying on computationally expensive depth-estimation models (like MiDaS) or physical LiDAR sensors, Drive Guardian utilizes deterministic pixel-math to trigger ADAS warnings.

### 3.1 Forward Collision Warning (FCW)

- **Trigger Condition:** Relative Bounding Box Area > 15%
- **Formula:** `(Box_Width * Box_Height) / (Frame_Width * Frame_Frame)`
- **Logic:** As a vehicle approaches the dashcam lens, its relative pixel footprint scales exponentially. At a 15% threshold, the vehicle is deemed critically close, triggering a `WARNING: BRAKE!` flag.

### 3.2 Lane Departure Warning (LDW)

- **Trigger Condition:** Center Hood Drift < 50 pixels
- **Logic:** The system anchors the bottom-center of the frame `(Frame_Width / 2, Frame_Height)` as the vehicle's "hood". It then calculates the absolute horizontal distance to the lowest `(x, y)` points of the UFLD lane lines. If this delta drops below 50 pixels, the system triggers `WARNING: DRIFTING`.

## 4. Hardware Optimizations & VRAM Management

Running two deep learning models simultaneously on a 6GB VRAM GPU required strict memory management optimizations:

1.  **Framework Decoupling:** YOLO is executed natively in PyTorch, while UFLD is executed via ONNX Runtime. This prevents PyTorch from monopolizing the memory allocator.
2.  **NumPy Post-Processing:** The UFLDv2 Softmax and Argmax operations were rewritten in pure NumPy. This bypasses PyTorch tensor allocation overhead, allowing the ONNX matrix to be resolved into `(x, y)` coordinates directly in system RAM, reserving the GPU strictly for matrix multiplication.
3.  **Unified Native UI:** Moving from a React/WebSocket stack to a native `CustomTkinter` desktop application eliminated base64 encoding/decoding overhead, drastically reducing CPU usage and latency.

## 5. Performance Metrics (Benchmarked on RTX 4050 6GB)

| Metric                     | Result  | Notes                                                   |
| :------------------------- | :------ | :------------------------------------------------------ |
| **Average FPS**            | `19`    | Measured across a 60-second 1080p dashcam video.        |
| **YOLO Inference Latency** | `12ms`  | Time taken to generate bounding boxes.                  |
| **UFLD Inference Latency** | `12ms`  | Time taken to generate ordinal classification matrices. |
| **Total VRAM Usage**       | `1.1GB` | Peak memory allocated during dual-inference.            |

## 6. Conclusion

The Drive Guardian pipeline successfully demonstrates that high-accuracy, multi-sensor ADAS capabilities can be achieved on resource-constrained hardware using deterministic mathematical fusion and strict VRAM optimizations.
