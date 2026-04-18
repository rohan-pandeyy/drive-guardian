# Drive Guardian

An open-source Advanced Driver Assistance System (ADAS) project, re-architectured for high scalability and adaptability from powerful backend servers to resource-constrained edge devices (like the Raspberry Pi).

## Architecture

- **Core**: Hardware abstraction layer & profile-based configuration loading.
- **Inference Engine**: Executes deep-learning logic (Object detection, segmentation) independent of the UI. Supported through YOLO / ONNX runtime.
- **Desktop App**: Native Python UI built with `CustomTkinter`.
- **Edge Communications**: IoT integrations for MQTT or CAN-bus reading.
- **ML Pipeline**: Experimental notebooks and scripts for model training and exporting.


## YOLOv8 Throughput Optimization

The PyTorch YOLO runner now supports two YOLOv8-focused performance toggles through profile YAML files:

- `YOLO_ENABLE_FUSE`: Fuses convolution + batch norm layers to reduce inference overhead.
- `YOLO_ENABLE_HALF`: Uses FP16 on CUDA for higher throughput.
- `YOLO_ENABLE_DCP_DEHAZE`: Applies Dark Channel Prior (DCP) dehazing before YOLO inference for fog/haze robustness.

Example profile values:

```yaml
YOLO_ENABLE_FUSE: true
YOLO_ENABLE_HALF: true
YOLO_ENABLE_DCP_DEHAZE: true
```

Notes:

- These options are applied when using a PyTorch `.pt` model via `YoloRunner`.
- ONNX runtime paths are not affected by these toggles.

### Dark Channel Prior (DCP) Dehazing

`DCP` is an image restoration method that estimates haze thickness from low-intensity local patches (the dark channel), estimates atmospheric light, and reconstructs a clearer scene before detection. In Drive Guardian this is implemented as a lightweight pre-processing function to improve object visibility in hazy conditions.
