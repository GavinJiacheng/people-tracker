# YOLO + StrongSORT Tracking (Dockerized, Modular Version)

This project provides a **modular, production-ready pipeline** for human detection and multi-object tracking using:

- **YOLO (Ultralytics)**
- **BoxMOT (StrongSORT + ReID)**
- **PyTorch (CUDA accelerated)**

The environment is fully containerized with Docker, ensuring reproducibility and zero local dependency conflicts.

---

## 🚀 Features

- **GPU Acceleration (CUDA 12.1)**  
  Optimized for NVIDIA GPUs (GTX / RTX series)

- **Modular Code Structure**  
  Clean separation of:
  - detection
  - tracking
  - visualization
  - video I/O

- **ReID-based Tracking (StrongSORT)**  
  Stable ID tracking with OSNet backbone

- **Dockerized Environment**  
  No need to manually install CUDA / PyTorch / dependencies

- **Flexible Model Switching**  
  Easily swap YOLO models (`n`, `s`, `m`, etc.)

---

## 📋 Prerequisites

Make sure the following are installed:

1. **NVIDIA Driver**
   - Version ≥ 528.33 (CUDA 12.1 compatible)

2. **Docker Desktop**
   - WSL2 backend enabled (Windows)

3. **NVIDIA Container Toolkit**
   - Required for GPU access inside Docker

---

## 📂 Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── detector.py
│   ├── tracker.py
│   ├── visualizer.py
│   ├── video_io.py
│   └── utils.py
│
├── yolov8n.pt                     # YOLO weights (REQUIRED)
├── osnet_ain_x1_0_msmt17.pt      # ReID weights (REQUIRED)
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚠️ Important Notes (Weights Placement)

The following files **must be placed in the same directory as the Dockerfile**:

- `yolov8n.pt`
- `osnet_ain_x1_0_msmt17.pt`

This is required because the Docker build process will copy them into the container.

---

## 🛠️ Build Docker Image

From project root:

```bash
docker build -t yolo-strongsort-app .
```

---

## 💻 Usage (Inference)

### Run with local video (recommended)

```bash
docker run --rm --gpus all -v "${PWD}:/data" yolo-strongsort-app python -m app.main --input /data/INPUT_FILE.mp4 --output /data/OUTPUT_FILE.mp4

```

---

## 📌 Parameter Explanation

- `--input`: Input video path (inside container)
- `--output`: Output video path (saved to host)
- `--model`: YOLO weights (default: yolov8n.pt)
- `--reid`: ReID weights (default: osnet)
- `--conf`: Detection confidence threshold
- `--imgsz`: Inference resolution

Example:

```bash
python -m app.main \
  --input /data/input.mp4 \
  --output /data/output.mp4 \
  --model yolov8m.pt \
  --conf 0.5 \
  --imgsz 1280
```

---

## ⚙️ How It Works

Pipeline per frame:

1. **YOLO Detection**
   - Filters only human class (`class=0`)
   - Outputs bounding boxes

2. **StrongSORT Tracking**
   - Associates detections across frames
   - Uses ReID embeddings (OSNet)
   - Maintains consistent track IDs

3. **Visualization**
   - Draws bounding boxes
   - Displays track IDs

4. **Video Output**
   - Encodes processed frames to output video

---

## 🧠 Design Choices

This project is built around a standard detect-then-track pipeline, with specific choices made to handle dense scenes and frequent occlusions.

### Detection: YOLOv8

A real-time object detector is required as the first stage. I selected **YOLOv8** from Ultralytics due to:

- strong performance-speed tradeoff
- ease of integration
- reliable bounding box outputs for tracking pipelines

The initial implementation used **YOLOv8m** for higher detection accuracy.

---

### Tracking: StrongSORT (with ReID)

The input videos contain:

- high object density  
- frequent occlusions  
- repeated re-appearance of individuals  

In such scenarios, motion-based tracking alone (e.g., IoU + Kalman filtering) is not sufficient.

Therefore, a tracker with **Re-identification (ReID)** capability is required.

I selected **StrongSORT**, which combines:

- motion modeling (Kalman filter)  
- appearance embedding (ReID via OSNet)  
- association logic for stable ID assignment  

Compared to methods like ByteTrack, which mainly rely on detection association, StrongSORT uses appearance features to recover identities after occlusion. This makes it more suitable for crowded scenes where people are frequently blocked and reappear.

Overall, StrongSORT provides better identity consistency in this project, where reducing ID switches is more important than maximizing speed.

---

### Model Adjustment: YOLOv8m → YOLOv8n

During testing, an important observation was made:

- **YOLOv8m**, while more accurate, produced bounding boxes that fluctuated noticeably frame-to-frame
- This instability negatively affected the ReID embeddings and tracking association
- As a result, it increased ID switching in practice

Switching to **YOLOv8n** improved overall tracking stability:

- slightly less precise boxes
- but significantly more stable over time
- leading to better tracking performance with StrongSORT

---

### Summary

- Detection accuracy alone is not sufficient for tracking quality
- Temporal stability of bounding boxes is critical for ReID-based trackers
- A lighter detector (YOLOv8n) can outperform a heavier one (YOLOv8m) in tracking scenarios due to reduced jitter

---

## 🐳 Docker Environment

- **Base Image**: `pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime`
- **Python**: 3.10
- **PyTorch**: 2.2.1+cu121
- **Ultralytics**: 8.x
- **BoxMOT**: 17.0.0

---

### Common Issues

#### 1. GPU Not Detected

- Ensure NVIDIA Container Toolkit is installed
- Run:

```bash
docker run --rm --gpus all yolo-strongsort-app nvidia-smi
```

---

#### 2. File Not Found

When using volume mount, always use `/data/` prefix inside container:

```bash
--input /data/input.mp4
--output /data/output.mp4
```

❌ Wrong (host path inside container):

```bash
--input input.mp4
```

---

#### 3. Wrong Mount Path

❌ DO NOT mount to `/app` (this overrides application code inside container):

```bash
-v "${PWD}:/app"   # ❌ WRONG
```

✔ Correct:

```bash
-v "${PWD}:/data"
```

---

#### 4. Model Not Found

Make sure these files exist in the same directory as `Dockerfile` **before building image**:

```text
yolov8n.pt
osnet_ain_x1_0_msmt17.pt
```

Then rebuild:

```bash
docker build -t yolo-strongsort-app .
```

---

#### 5. Input file not found inside container

If you see errors like:

```text
FileNotFoundError: /data/xxx.mp4
```

Make sure:

- The file exists in your current folder
- You mounted the folder correctly:

```bash
-v "${PWD}:/data"
```

---

## 📄 License

This project depends on:

- Ultralytics YOLO
- BoxMOT

Please follow their respective licenses.

