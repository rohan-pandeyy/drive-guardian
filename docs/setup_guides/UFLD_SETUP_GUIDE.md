# Downloading and Installing Custom UFLDv2 Models

Unlike the YOLO object detection weights that pull down automatically from the Ultralytics GitHub servers during runtime, the UFLD (Ultra-Fast-Lane-Detection) lane detection engine relies on a strictly defined `ufldv2_tusimple_res18.onnx` file.

Because these weights are uniquely converted from PyTorch (`.pth`) to ONNX (`.onnx` / `.onnx.data`) execution graphs for optimal performance within this specific dual-inference architecture, they are currently hosted privately.

## Step 1: Download the Model Files

You need to download the ONNX weights from the Google Drive link provided below.

**UFLDv2 ONNX:** [Link](https://drive.google.com/drive/folders/17i1_yf3PO3whdueNeKeWw6bydRWu4HmB?usp=sharing)

You will need to download the following files:

- `ufldv2_tusimple_res18.onnx`

If your configuration utilizes the `.pth` PyTorch fallback, or the specific ONNX graph relies on separated `.data` weights, download those as well:

- `ufldv2_tusimple_res18.onnx.data` (if available/required)
- `ufldv2_tusimple_res18.pth` (fallback only)

## Step 2: Place the Files in the Repository

Once downloaded, you must place these files into the `models/` directory at the root of the project.

Your directory structure MUST look identically to this before running the application:

```text
Drive-Guardian/
├── core/
├── desktop_app/
├── inference_engine/
└── models/
    ├── yolov12n.pt
    ├── ufldv2_tusimple_res18.onnx   <-- PLACE HERE
    └── tusimple_res18.onnx.data <-- PLACE HERE
```

## Step 3: Run the Application

Once the files are located in the `models/` directory, the application will automatically hook them during startup.

In your terminal, activate your virtual environment and run the desktop app:

```bash
.\.venv\Scripts\Activate.ps1
python desktop_app/main.py
```

You should see the following logs verifying successful deployment:

```text
[INFO] Loading UFLD Lane Detection model from models/ufldv2_tusimple_res18.onnx...
[INFO] UFLD Runner initialized using cuda:0
```
