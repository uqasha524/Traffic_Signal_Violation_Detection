<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:0d0221,40:ff0040,100:ff8800&height=220&section=header&text=Traffic%20Violation%20Detection&fontSize=42&fontColor=ffffff&fontAlignY=45&desc=🚦%20AI-Powered%20Red-Light%20Violation%20Detection%20System&descSize=20&descAlignY=70&animation=fadeIn" width="100%"/>

</div>

<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=18&duration=3000&pause=800&color=FF4500&center=true&vCenter=true&width=800&lines=Autonomous+Red-Light+Violation+Detection+🚗;Multi-Lane+Vehicle+Tracking+with+Unique+IDs;Auto-Generated+Timestamped+Evidence+Reports;YOLO+%2B+DeepSort+%2B+OpenCV+—+Real-Time+AI)](https://git.io/typing-svg)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo & Screenshots](#-demo--screenshots)
- [System Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Pipeline Breakdown](#-pipeline-breakdown)
- [ROI Selection](#-roi-selection--zebra-crossing-polygon)
- [Violation Detection Logic](#-violation-detection-logic)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
- [Technical Details](#-technical-details)
- [Requirements](#-requirements)
- [Connect With Me](#-connect-with-me)

---

## 🎯 Overview

A **production-grade, autonomous traffic surveillance system** that detects and records vehicles running red lights across multiple lanes in real time. The system uses dual **YOLO** deep learning models for traffic light state classification and vehicle detection, combined with **DeepSort multi-object tracking** to assign persistent unique IDs to every vehicle — even during occlusions.

When a vehicle is detected crossing the designated zebra crossing zone during a red signal, the system **automatically captures a timestamped screenshot** as evidence, labeling the vehicle's unique ID.

> **No human operator needed — the AI watches, tracks, and records automatically.**

### 🔑 Key Highlights

| Feature | Detail |
|:---|:---|
| 🚦 **Traffic Light AI** | Custom-trained YOLO model — detects Red, Green, RedRight signals |
| 🚗 **Vehicle Detection** | Custom YOLO model with confidence threshold 0.15 for dense traffic |
| 🔁 **Multi-Object Tracking** | DeepSort tracker — persistent IDs across frames & occlusions |
| 📐 **ROI** | User-defined polygon zone (zebra crossing) via mouse click |
| 📸 **Evidence** | Auto-saved violation screenshots per unique vehicle ID |
| 🛣️ **Multi-Lane** | Simultaneous tracking of all vehicles across all lanes |
| ⚡ **Real-Time** | Frame-by-frame processing at video playback speed |

---

## 🖼️ Demo & Screenshots

> Add your demo screenshots in the `assets/` folder and they will appear here.

### 📸 Violation Evidence Screenshots
*The system automatically saves evidence like this for every detected violation:*

```
Violation_ScreenShots/
├── violation_1.jpg    ← Vehicle ID 1 caught at red light
├── violation_7.jpg    ← Vehicle ID 7 caught at red light
├── violation_23.jpg   ← Vehicle ID 23 caught at red light
└── ...
```

Each screenshot contains:
- Bounding box around the violating vehicle in **red**
- Text overlay: `"ID {track_id} - VIOLATED"`
- Full frame context for evidence

### 🎬 Live Detection Preview
```
┌──────────────────────────────────────────────────────┐
│  🔴 Red Light Detected                               │
│                                                      │
│  Violators: ID 3, ID 7                               │
│                                                      │
│  ┌────────┐  ┌────────┐  ┌──────────────┐           │
│  │ ID: 1  │  │ ID: 3  │  │   ID: 7      │           │
│  │ GREEN  │  │  RED   │  │    RED       │           │
│  └────────┘  └────────┘  └──────────────┘           │
│                                                      │
│  ════════ [  ZEBRA CROSSING ROI  ] ══════════        │
└──────────────────────────────────────────────────────┘
```

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║            TRAFFIC VIOLATION DETECTION PIPELINE                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📹 Video Input (CCTV / Dashcam footage)                            ║
║       │                                                              ║
║       ├──────────────────────────────────────────────────┐          ║
║       │                                                  │          ║
║       ▼                                                  ▼          ║
║  🚦 YOLO Traffic Light Model            📐 Polygon ROI (Zebra)     ║
║     │ conf=0.5                               (User-defined)         ║
║     │                                                               ║
║     ▼                                                               ║
║  Red Light?                                                          ║
║     │ YES                                                            ║
║     ▼                                                                ║
║  🚗 YOLO Vehicle Detection (conf=0.15)                              ║
║     │                                                                ║
║     ▼                                                                ║
║  🔁 DeepSort Multi-Object Tracker                                   ║
║     │   max_age=70, n_init=5, embedder=mobilenet                    ║
║     │                                                                ║
║     ▼                                                                ║
║  📐 Polygon Intersection Test (pointPolygonTest)                    ║
║     │   vehicle center point vs zebra polygon                       ║
║     │                                                                ║
║     ├── INSIDE polygon?                                             ║
║     │       │ YES + new ID?                                         ║
║     │       ▼                                                        ║
║     │   📸 Save Violation Screenshot                                ║
║     │   🔴 Mark ID as Violator                                      ║
║     │   🖥️  Display on screen                                        ║
║     │                                                                ║
║     └── OUTSIDE → Normal tracking (green box)                      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## ⚙️ How It Works

### Step 1 — Select the Zebra Crossing Zone
The user clicks **4 points** on the first video frame to define the violation zone (zebra crossing / stop line). The system draws a polygon and asks for confirmation before processing begins.

### Step 2 — Traffic Light Monitoring
Every frame is analyzed by the **Traffic Light YOLO model**. The system checks for:
- `Red` → violation monitoring **ACTIVE** 🔴
- `RedRight` → violation monitoring **ACTIVE** 🔴
- `Green` → vehicle tracking **PAUSED** 🟢 (no violation possible)

### Step 3 — Vehicle Detection & Tracking (Red Light Only)
When a red light is detected, the **Vehicle YOLO model** detects all vehicles. These detections are passed to **DeepSort**, which:
- Assigns persistent unique IDs to each vehicle
- Maintains IDs across frames using MobileNet appearance embeddings
- Handles partial occlusions with IoU-based re-identification

### Step 4 — Violation Check
For each tracked vehicle, the system checks if the vehicle's **center point** falls inside the user-defined polygon using OpenCV's `pointPolygonTest`. If it does during a red light → **VIOLATION**.

### Step 5 — Evidence Generation
First-time violators (new IDs crossing for the first time) trigger:
- Screenshot saved to `Violation_ScreenShots/violation_{ID}.jpg`
- Red bounding box on screen
- ID added to permanent `violated_ids` set
- Violator list displayed in top-left of live feed

---

## ✨ Features

| Feature | Description |
|:---|:---|
| 🎯 **Dual YOLO Models** | Separate specialized models for traffic light and vehicle detection |
| 📍 **Interactive ROI** | Click-to-define polygon zone — adaptable to any camera angle |
| 🔢 **Persistent IDs** | Same vehicle keeps same ID even after occlusion |
| 📸 **Auto Evidence** | No human needed — violations captured and saved automatically |
| 🔴 **State-Aware** | Vehicle tracking only activates during red light — no false positives |
| 🛣️ **Multi-Lane** | Handles multiple vehicles across multiple lanes simultaneously |
| 🖥️ **Live Overlay** | Real-time bounding boxes, IDs, signal state, violator list |
| 🔄 **Robust Tracking** | `max_age=70` frames tolerance — handles momentary detection gaps |

---

## 🔬 Pipeline Breakdown

### 1. Zebra Crossing ROI Selection

```python
# User clicks 4 points on first video frame
# System maps display coordinates → original video coordinates
orig_x = int(x / scale_x)   # reverse scaling
orig_y = int(y / scale_y)

# On 4th click → polygon is closed
# Press 'c' → confirm | Press 'a' → reset and re-draw
```

### 2. Traffic Light Detection

```python
traffic_results = traffic_light_model.predict(frame, conf=0.5)

# Labels checked:
if label in ['Red', 'RedRight']:
    red_light = True   # Activate violation monitoring
```

### 3. Vehicle Detection + DeepSort Tracking

```python
# YOLO detections → DeepSort format: [x, y, w, h]
detections.append(([x1, y1, x2-x1, y2-y1], conf, "vehicle"))

# DeepSort assigns persistent Track IDs
tracks = tracker.update_tracks(detections, frame=frame)
```

DeepSort Configuration:
```python
tracker = DeepSort(
    max_age=70,           # Keep track alive for 70 frames without detection
    n_init=5,             # Confirm track after 5 consecutive detections
    max_iou_distance=0.5, # IoU threshold for track association
    embedder="mobilenet"  # Appearance feature extractor
)
```

### 4. Violation Detection Logic

```python
def is_vehicle_crossing(box, polygon_points):
    x_center = int((box[0] + box[2]) / 2)
    y_center = int((box[1] + box[3]) / 2)
    # Returns True if vehicle center is inside polygon
    return cv2.pointPolygonTest(polygon_points, (x_center, y_center), False) >= 0
```

### 5. Evidence Screenshot

```python
if is_vehicle_crossing(...) and track_id not in violated_ids:
    violated_ids.add(track_id)   # Mark as violated (only capture once per vehicle)
    ss = frame.copy()
    cv2.rectangle(ss, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(ss, f"ID {track_id} - VIOLATED", ...)
    cv2.imwrite(f"Violation_ScreenShots/violation_{track_id}.jpg", ss)
```

---

## 📁 Project Structure

```
Traffic_Signal_Violation_Detection/
│
├── 📂 NoteBook/
│   └── main.py                          # Main pipeline script
│
├── 📂 Models/
│   ├── vehicle_detection.pt             # Custom YOLO vehicle detection model
│   └── Traffic_Light_Detection.pt       # Custom YOLO traffic light model
│
├── 📂 Violation_ScreenShots/            # Auto-generated evidence folder
│   ├── violation_1.jpg
│   ├── violation_7.jpg
│   └── ...
│
├── 📂 assets/                           # Demo images / GIFs for README
│   └── demo.gif
│
└── README.md
```

---

## 🛠️ Installation & Setup

### Prerequisites

- Python **3.9+**
- GPU recommended (CUDA) for real-time performance
- CCTV / dashcam video file for testing

### 1. Clone the Repository

```bash
git clone https://github.com/uqasha524/Traffic_Signal_Violation_Detection.git
cd Traffic_Signal_Violation_Detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
pip install numpy
```

Or install everything at once:

```bash
pip install ultralytics deep-sort-realtime opencv-python numpy
```

### 4. Add Model Files

Place your trained YOLO models inside the `Models/` folder:

```
Models/
├── vehicle_detection.pt
└── Traffic_Light_Detection.pt
```

> ⚠️ Models are not included in this repository due to file size. Train your own or contact the author.

---

## 🚀 How to Run

### Basic Usage

```bash
cd NoteBook
python main.py
```

### Update the Video Path

Open `main.py` and update this line with your video file path:

```python
video_path = r"path/to/your/video.mp4"
```

### Runtime Controls

| Step | Action |
|:---|:---|
| **1** | Window opens showing first video frame |
| **2** | Click **4 points** to define the zebra crossing polygon |
| **3** | Press **`C`** to confirm the zone |
| **4** | Press **`A`** to reset and redraw if unhappy |
| **5** | Press **`Q`** to quit at any time |

### Expected Output

```
Screenshot saved: .\Violation_ScreenShots\violation_3.jpg
Screenshot saved: .\Violation_ScreenShots\violation_7.jpg
Screenshot saved: .\Violation_ScreenShots\violation_15.jpg
```

- Live video window shows bounding boxes + IDs
- Red boxes = confirmed violators
- Green boxes = non-violating tracked vehicles
- Top-left overlay shows current violator IDs in real time

---

## 🔧 Technical Details

### Why `conf=0.15` for Vehicle Detection?

Dense urban traffic scenes have many partially occluded vehicles. A lower confidence threshold ensures:
- Partially visible vehicles are detected early
- Vehicles at frame edges are captured
- No missed violations due to occlusion

The tracking layer (`n_init=5`) acts as the quality filter — only vehicles confirmed across 5 frames are assigned IDs, preventing false positive tracks.

### Why DeepSort over ByteTrack?

DeepSort uses **MobileNet appearance embeddings** alongside IoU matching. This means:
- Vehicles that temporarily leave the frame (overtaken, occluded) are **re-identified** using visual features
- Same vehicle gets the **same ID** even after disappearing for up to **70 frames**
- Critical for multi-lane scenarios with crossing paths

### Polygon Test (OpenCV)

```python
cv2.pointPolygonTest(polygon, point, measureDist=False)
# Returns: >= 0 → inside or on border (VIOLATION)
#          <  0 → outside (safe)
```

Using the **center point** of the bounding box rather than corners prevents false triggers from vehicles adjacent to the zone.

### Scale-Aware ROI Selection

The system displays video at 1280×720 for user interaction, but internally maps all clicks back to original video resolution:

```python
scale_x = display_width / orig_width
scale_y = display_height / orig_height

# Reverse mapping on click
orig_x = int(x / scale_x)
orig_y = int(y / scale_y)
```

This ensures the polygon is accurate regardless of display resolution.

---

## 📦 Requirements

```
ultralytics>=8.0
deep-sort-realtime>=1.3
opencv-python>=4.8
numpy>=1.24
```

---

## 🔮 Future Improvements

- [ ] Add **license plate OCR** for vehicle identification
- [ ] Integrate **timestamp overlay** on violation screenshots
- [ ] Export **violation report as PDF/CSV** with ID, time, frame number
- [ ] Add support for **live RTSP camera streams**
- [ ] Add **speed estimation** alongside red-light violation
- [ ] Web dashboard for viewing all violations in real time
- [ ] Multi-camera support for intersection coverage

---

## 🔗 Connect With Me

<div align="center">

[![Email](https://img.shields.io/badge/Gmail-uqashazahid%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:uqashazahid@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Uqasha%20Zahid-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/uqasha-zahid)
[![GitHub](https://img.shields.io/badge/GitHub-uqasha524-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/uqasha524)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Uqasha-FFD21E?style=for-the-badge)](https://huggingface.co/Uqasha)

</div>

<div align="center">

### 🚀 More Projects

| Project | Description | Link |
|:---:|:---:|:---:|
| ⚡ PowerGuard AI | Electricity Theft Detection — Live SCADA System | [🤗 Live Demo](https://huggingface.co/spaces/Uqasha/FYP-Electricity_Theft_Detection) |
| 🕹️ Action Recognition | Real-Time Gaming Interface via Body Gestures | [GitHub](https://github.com/uqasha524) |
| 👁️ Facial Attendance | AI Biometric Attendance for 100+ Users | [GitHub](https://github.com/uqasha524) |

</div>

---

<div align="center">

**Built with ❤️ using Python · YOLO · DeepSort · OpenCV**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=flat-square&logo=OpenCV&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-111111?style=flat-square&logo=yolo&logoColor=00FFFF)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ff0040,100:ff8800&height=100&section=footer" width="100%"/>

</div>
