# Contribution: ENABLE_TRACKING Integration

**Contributor:** Yousuf  
**Date:** March 2026  
**Module affected:** `inference_engine/detectors/yolo_runner.py`

---

## What Was the Problem?

The project's configuration system (`core/config.py`) loads a hardware profile
at startup — either `high_power.yaml` (GPU workstation) or `lite_edge.yaml`
(resource-constrained edge device). Both profiles define an `ENABLE_TRACKING`
flag:

```yaml
# high_power.yaml
ENABLE_TRACKING: true

# lite_edge.yaml
ENABLE_TRACKING: false
```

Despite being loaded into the `Settings` object, this flag was **never read or
applied anywhere in the codebase**. The YOLO object detector always ran in plain
detection mode regardless of the active profile, making the config key a dead
stub with no effect.

---

## What Was Changed?

### 1. `inference_engine/detectors/yolo_runner.py` — Core change

**Before:**

```python
import cv2                          # unused — dead import
import torch
from ultralytics import YOLO
from .object_detector import ObjectDetector

# ...

def process_frame(self, frame):
    if self.model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    results = self.model(frame, device=self.device, verbose=False)
    return results[0]
```

**After:**

```python
import torch
from ultralytics import YOLO
from .object_detector import ObjectDetector
from core.config import settings       # read ENABLE_TRACKING and TRACKER at call time

# ...

def process_frame(self, frame):
    if self.model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    if settings.ENABLE_TRACKING:
        # Tracking mode: each detected object gets a persistent integer ID
        # across frames. persist=True is required to maintain tracker state
        # between calls in a frame loop — without it IDs reset every frame.
        results = self.model.track(
            frame,
            persist=True,
            tracker=settings.TRACKER,
            device=self.device,
            verbose=False,
        )
    else:
        # Detection-only mode: faster, stateless, used on edge devices
        results = self.model(frame, device=self.device, verbose=False)

    return results[0]
```

Additionally, the unused `import cv2` dead import was removed.

---

### 2. `core/config.py` — Added `TRACKER` field to `Settings`

Added a `TRACKER` field (default `"bytetrack.yaml"`) to the `Settings` dataclass
so the tracker algorithm is configurable per profile without touching code.

---

### 3. `core/profiles/high_power.yaml` — Added `TRACKER` key

```yaml
TRACKER: 'bytetrack.yaml'
```

This makes the tracker algorithm an explicit, documented configuration option
for the high-power profile. The `lite_edge` profile needs no change because
`ENABLE_TRACKING: false` skips the tracking path entirely.

---

## Why `persist=True`?

Ultralytics tracking is stateful. Without `persist=True`, the tracker resets
its internal state on every frame call, assigning new IDs each time — making
tracking functionally useless in a video loop. `persist=True` tells the tracker
that each call is a continuation of the same stream, enabling stable
cross-frame object identities.

---

## Why ByteTrack Over BoT-SORT (the Ultralytics default)?

| Property | BoT-SORT (default) | ByteTrack (chosen) |
|---|---|---|
| ReID appearance model | Optional (loads extra weights into VRAM) | None required |
| VRAM overhead | Higher | Minimal |
| Real-time suitability | Moderate | High |
| Occlusion recovery | Strong (with ReID) | Good |

Drive Guardian targets real-time inference at ~19 FPS on an RTX 4050 6 GB
(~1.1 GB VRAM budget, documented in `docs/experimental_results.md`).
ByteTrack adds negligible overhead versus BoT-SORT's optional ReID feature
extractor, making it the safer default for an ADAS real-time pipeline.
The tracker remains swappable via the `TRACKER` config key.

---

## Impact

| Before this contribution | After this contribution |
|---|---|
| `ENABLE_TRACKING: true` in the profile had zero effect on runtime behaviour | `high_power` profile enables full multi-object tracking with persistent IDs |
| Every frame was processed by a stateless detector | Each detected vehicle receives a stable integer ID across frames |
| No track IDs visible in the desktop UI | `Results.plot()` (already called in `video_thread.py`) auto-renders IDs — zero UI changes required |
| `ENABLE_TRACKING: false` was silently ignored | `lite_edge` profile correctly stays in lightweight detection-only mode |
| `TRACKER` setting did not exist | Tracker algorithm is now configurable per profile via YAML |

The collision warning (`ForwardCollisionWarning`) and lane departure warning
(`LaneDepartureWarning`) downstream consumers are completely unaffected — they
operate on the same `Results` object whether tracking is active or not.

---

## Files Modified

| File | Type of change |
|---|---|
| `inference_engine/detectors/yolo_runner.py` | Added tracking branch in `process_frame()`, removed dead `import cv2` |
| `core/config.py` | Added `TRACKER: str` field to `Settings` and its `load_profile()` mapping |
| `core/profiles/high_power.yaml` | Added `TRACKER: 'bytetrack.yaml'` |
| `core/profiles/lite_edge.yaml` | No change (tracking path is not taken when `ENABLE_TRACKING: false`) |

---

## How to Verify

```bash
# Tracking ON (high_power profile)
DRIVE_GUARDIAN_PROFILE=high_power python -m desktop_app.main
# Expected: bounding boxes in the UI show integer track IDs, e.g. "car [3]"

# Tracking OFF (lite_edge profile)
DRIVE_GUARDIAN_PROFILE=lite_edge python -m desktop_app.main
# Expected: plain bounding boxes with no IDs — identical to original behaviour
```
