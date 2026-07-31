# Attendance System

> Technical Documentation & User Guide

**Built with:** InsightFace · Flask · OpenCV · Python  
**Version:** 1.0 (2025)

## 1. System Overview

The Attendance System is a locally hosted web application that automatically marks student attendance using face recognition. A teacher starts a session, points the classroom camera at students, and recognized students are marked **Present** automatically.

- Runs entirely on the local computer
- No internet required after the initial model download
- Student records, photos, and attendance reports are stored locally

## Key Features

- Real-time face detection and recognition
- Automatic attendance marking
- Manual attendance override
- Student registry (Roll No., Name, Course, Year, Section)
- Live photo capture
- Session management
- CSV attendance export
- Organized `data/` directory

## 2. Project Structure

```text
attendance_app/
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
└── data/
    ├── db.json
    ├── students/
    └── attendance/
```

## 3. System Architecture

### Face Recognition
1. Detect faces with InsightFace.
2. Generate 512-dimensional face embeddings.
3. Compare embeddings using cosine similarity.
4. Mark matching students present.

Recognition runs every 8th frame by default to improve performance.

### Attendance
- Each student is marked only once per session.
- Manual overrides are supported.
- Attendance refreshes automatically.

## 4. Why InsightFace?

InsightFace was selected because it provides:

- High recognition accuracy
- Simple installation
- Fast CPU performance
- ONNX Runtime support
- No TensorFlow dependency

Alternative libraries discussed in the original documentation include face_recognition (dlib), DeepFace, OpenCV Haar Cascades, and FaceNet.

## 5. Installation

### Requirements

- Windows 10/11, Ubuntu 20.04+, or macOS 12+
- Python 3.8–3.11
- Webcam
- ~500 MB free disk space

### Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install flask flask-cors opencv-python numpy insightface onnxruntime pillow
```

### Run

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

## 6. User Guide

1. Add classes and teachers.
2. Register students.
3. Capture 4–6 photos per student.
4. Start a live session.
5. Export attendance as CSV.

## 7. Challenges Solved

- Browser camera permission restrictions
- Windows installation issues with dlib
- CPU performance optimization
- Duplicate attendance prevention
- Organized data storage
- TensorFlow dependency conflicts

## 8. FAQ

The original document includes answers covering:

- Multiple student recognition
- Unknown faces
- Recommended number of photos
- Recognition threshold
- Lighting
- Multiple cameras
- Offline usage
- GPU support
- YOLO vs InsightFace

## 9. Configuration

Important settings include:

- `RECOG_EVERY`
- Recognition threshold
- Detection size
- Camera resolution
- JPEG quality

## 10. Troubleshooting

Common issues covered:

- Camera not working
- Students recognized as Unknown
- High CPU usage
- Model download problems
- Port conflicts
- Photos not saving
- Incorrect face matches

---
For complete implementation details, explanations, comparison tables, and configuration values, refer to the original technical documentation.
