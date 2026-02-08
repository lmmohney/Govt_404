# Govt_404 - Live Webcam Image Tracker

AI Tool Demo: Real-time webcam object detection using YOLOv8 (completely free, no API key needed).

## Features
- **Live Webcam Access**: Real-time video feed from your computer's webcam
- **Free AI-Powered Detection**: Uses YOLOv8 (open-source computer vision model)
- **Runs Locally**: No API calls, no costs, all processing on your computer
- **Real-Time Bounding Boxes**: Shows detected objects with confidence scores
- **Fast & Efficient**: Nano model optimized for real-time performance

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- OpenCV (webcam access)
- YOLOv8 from Ultralytics (object detection)
- PyTorch (ML framework)

### 2. Run the Tracker
```bash
python main.py
```

The first run will download the YOLOv8 model (~45MB). No additional setup required!

Press `q` to quit the application.

## How It Works
1. Captures video frames from your webcam
2. Processes every 2 frames using YOLOv8 for speed
3. Detects objects with bounding boxes and confidence scores
4. Displays results on-screen and in the console

## Configuration
In `main.py`, you can adjust:
- `model_name`: Change to 'yolov8s.pt' or 'yolov8m.pt' for better accuracy (slower)
- `process_every_n_frames`: Lower = more frequent updates, higher = faster performance
- Confidence threshold in `identify_objects()` method

## Model Options
- **yolov8n.pt** (Nano) - Fastest, ~45MB
- **yolov8s.pt** (Small) - Balanced, ~90MB
- **yolov8m.pt** (Medium) - Better accuracy, ~180MB
- **yolov8l.pt** (Large) - High accuracy, ~400MB

## Requirements
- Python 3.8+
- Webcam access
- ~500MB disk space for model files
- No internet required after initial setup 
