# ByteTrack Algorithm in Drive Guardian

Drive Guardian relies on **ByteTrack** (bundled via Ultralytics) to assign persistent tracking IDs to vehicles detected by YOLO. This prevents the ADAS logic from treating a single vehicle as 30 different vehicles across 30 consecutive video frames.

While YOLO is very robust at detecting vehicles on the far horizon (often returning confidence scores around `0.35` to `0.45`), you might notice that ByteTrack **refuses to assign them an ID** until they get significantly closer to the dashcam.

This is not a bug; it is an intentional anti-ghosting mechanism built into ByteTrack's core mathematical logic.

## 1. How ByteTrack Works: The Two-Stage Association

ByteTrack solves tracking by splitting YOLO's bounding boxes into two separate pools based on their confidence scores, and attempting to match them to existing tracks using a Kalman filter.

1. **High-Confidence Detections (The "Sure Things"):** Boxes with confidence scores > `track_high_thresh` (default: `0.5`).
2. **Low-Confidence Detections (The "Maybes"):** Boxes with confidence scores between `track_low_thresh` (default: `0.1`) and `track_high_thresh`.

**Stage 1:** The algorithm first takes all High-Confidence detections and tries to match them to previously established vehicle tracks using Intersection over Union (IoU) overlaps.

**Stage 2:** If a track from a previous frame goes missing (e.g., a car drives behind a tree, causing its YOLO confidence to drop to `0.2`), ByteTrack uses the Low-Confidence detections to "recover" that missing track.

### The New Track Problem (Why Distant Cars Have No IDs)

This brings us to the core issue with distant vehicles: **ByteTrack will never initialize a brand new tracking ID from a Low-Confidence detection.**

According to the default `bytetrack.yaml` configuration:

- `track_high_thresh: 0.5`
- `new_track_thresh: 0.6`

If a car appears on the horizon for the very first time and YOLO gives it a confidence of `0.4`, it is placed into the Low-Confidence pool. Since this car has never been tracked before, there is no existing ID to recover in Stage 2. Furthermore, since its confidence is below `0.6`, the algorithm mathematically refuses to create a new ID for it.

The algorithm stubbornly waits until the car drives extremely close to the dashcam (thereby boosting its YOLO confidence past `0.6`) before it finally grants it a persistent integer ID.

## 2. Why is it Designed This Way?

This strict behavior prevents **ID Ghosting**.

In dense, noisy environments, YOLO will occasionally hallucinate a bounding box on a shadow, a signpost, or a scratch on the windshield, usually assigning it a very low confidence (`0.2`).

If ByteTrack allowed Low-Confidence detections to spawn new tracking IDs, every single shadow or false-positive glitch would generate a permanent ghost ID that the tracker would try to follow for the next 30 frames (controlled by the `track_buffer` parameter). This would rapidly consume CPU resources and clutter the ADAS logic pool with phantom threats.

By forcing a threshold of `0.6` for new tracks, ByteTrack guarantees that it is only locking onto physical, tangible vehicles. Once it "locks on" to a real vehicle, it is then perfectly willing to use those Low-Confidence `0.2` scores to hold onto the track if the car gets partially obscured.

## 3. Configuration Overrides

If an ADAS implementation desperately requires long-range tracking capabilities, these thresholds can be overridden by supplying a custom YAML configuration to the `model.track()` method:

```yaml
# custom_bytetrack.yaml prototype
tracker_type: bytetrack
track_high_thresh: 0.3 # Accept distant cars into the High pool
track_low_thresh: 0.1
new_track_thresh: 0.35 # Spawn new IDs for distant cars
track_buffer: 30
match_thresh: 0.8
fuse_score: True # Required by Ultralytics parser
```

However, doing so drastically increases the risk of the system generating persistent false-positive Forward Collision Warnings if the camera suffers from glare or windshield artifacts. Keeping the default strict thresholds ensures that only genuine, undeniable threats are tracked by the ego-vehicle.
