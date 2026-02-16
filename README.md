# Drive Guardian

An open-source Advanced Driver Assistance System (ADAS) project, re-architectured for high scalability and adaptability from powerful backend servers to resource-constrained edge devices (like the Raspberry Pi).

## Architecture

- **Core**: Hardware abstraction layer & profile-based configuration loading.
- **Inference Engine**: Executes deep-learning logic (Object detection, segmentation) independent of the UI. Supported through YOLO / ONNX runtime.
- **Desktop App**: Native Python UI built with `CustomTkinter`.
- **Edge Communications**: IoT integrations for MQTT or CAN-bus reading.
- **ML Pipeline**: Experimental notebooks and scripts for model training and exporting.
