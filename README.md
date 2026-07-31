
ATTENDANCE SYSTEM
Technical Documentation & User Guide



Built with InsightFace · Flask · OpenCV · Python

Version 1.0  ·  2025
1. System Overview
The Attendance System is a locally-run, web-based application that uses face recognition to automatically detect and mark student attendance in real time. A teacher starts a session, points the classroom camera at students as they enter, and the system identifies each student and marks them Present without any manual input.

The system runs entirely on your local PC — no cloud, no internet connection required for operation (only for the initial model download). All data including student photos, records, and attendance CSVs are stored locally in an organized folder structure.

Key Features
⦁	Real-time face detection and recognition via webcam
⦁	Automatic attendance marking — recognized students are instantly marked Present
⦁	Manual override — any student can be toggled Present/Absent using the toggle switch
⦁	Student registry with Roll No., Name, Course, Year, and Section fields
⦁	Photo capture modal with live preview and a thumbnail review strip
⦁	Session management with Teacher, Subject, and Class/Section fields
⦁	Attendance export as a structured CSV file
⦁	Clean data folder structure — all data lives inside data/ away from source code


2. Project Structure
The project is organized to keep source code separate from data. Everything generated at runtime lives inside the data/ folder.

attendance_app/
├── app.py                    ← Flask backend (main server)
├── requirements.txt          ← Python dependencies
├── templates/
│   └── index.html            ← Full web UI (single file)
└── data/                     ← All runtime data (auto-created)
    ├── db.json               ← Students, teachers, classes
    ├── students/             ← One folder per student
    │   └── <roll_no>/
    │       └── photos/
    │           ├── 20250225_143022_123456.jpg
    │           └── 20250225_143100_789012.jpg
    └── attendance/           ← Exported CSV files
        └── attendance_CS101_2025-02-25.csv

Why this structure? Keeping data/ separate means you can back up all student records and attendance history by copying a single folder. It also means you can wipe and redeploy the code without touching any data.

Photo filenames are timestamped (e.g. 20250225_143022_123456.jpg) so they are always unique, never overwrite each other, and are naturally sorted in chronological order.


3. How the System Works
3.1 Face Recognition Pipeline
The system uses a two-stage pipeline for every video frame:

1.	Detection — InsightFace locates all faces in the frame and returns bounding boxes.
2.	Recognition — For each detected face, a 512-dimensional embedding vector is extracted. This vector is compared against all stored student embeddings using cosine similarity. If the best match exceeds the threshold (0.4), the student is identified.

To keep the UI smooth, recognition only runs every 8th frame (configurable via RECOG_EVERY in app.py). Bounding boxes from the last recognition pass are reused for the intervening frames, which gives a fluid live view without maxing out the CPU.

3.2 Face Embeddings
When you capture photos of a student, each photo is processed by InsightFace to extract a 512-dimensional embedding — a compact numerical representation of that face. These embeddings are stored in memory at session start.

At recognition time, the dot product between the live face embedding and each stored embedding gives a cosine similarity score (since all embeddings are L2-normalised). A score of 1.0 means identical, 0.0 means unrelated. The threshold of 0.4 was chosen as a practical balance between rejecting impostors and accepting the same person under different lighting or angles.

More photos = better accuracy. Each student should ideally have 4-6 photos taken in the same room and lighting conditions as the classroom, from slightly different angles.

3.3 Attendance Marking
Once a student is identified in a frame, their roll number is added to the marked_today set. This is a Python set, so each student can only be marked once per session regardless of how many frames they appear in. The set resets to empty when a new session is started.

Manual overrides work by calling the /api/session/mark endpoint which toggles the roll number in or out of marked_today. The attendance panel in the UI refreshes every 2.5 seconds to reflect the current state.

3.4 Camera Feed
The live video is served as an MJPEG stream from the Flask server (/video_feed). The browser displays it as a simple <img> tag pointing to that URL. This approach has an important advantage: since the video comes from the server (localhost), the browser never asks for camera permission — the getUserMedia security restriction that caused issues in earlier versions is completely bypassed.

The same approach is used for the photo capture modal. A separate /capture_feed endpoint streams the camera with a simple Haar Cascade overlay for framing guidance. Clicking Capture hits the /api/students/<roll>/capture endpoint which grabs a frame server-side and saves it directly.


4. Why InsightFace? Model Comparison
Choosing the right face recognition library is critical. Here is a comparison of the major options and why InsightFace was selected.

Library	Accuracy	Windows Install	Speed (CPU)	Active?	Verdict
face_recognition (dlib)	Good	Very painful (cmake + dlib)	Moderate	No (2020)	Rejected
DeepFace	Very Good	Requires TensorFlow	Slow	Yes	Rejected
OpenCV Haar Cascade	Low	Built-in	Very Fast	Yes	Detection only
InsightFace + ONNX	Excellent	pip install only	Fast	Yes	Selected
FaceNet (PyTorch)	Excellent	Moderate	Moderate	Partial	Alternative

4.1 face_recognition (dlib)
This was the original choice but was rejected due to its installation complexity on Windows. It requires cmake and the Visual C++ Build Tools to compile dlib from source. The library itself has not been meaningfully updated since 2020 and uses an older ResNet model. Even when installed correctly, version conflicts between dlib 20.x and face_recognition 1.3.0 caused repeated failures.

4.2 DeepFace
DeepFace is a wrapper around multiple models (Facenet, VGG-Face, ArcFace, etc.) and is actively maintained. However it has a hard dependency on TensorFlow, which has its own heavy installation requirements and version conflicts. On Python 3.12+ TensorFlow does not install cleanly at all. This made DeepFace unsuitable for a simple local setup.

4.3 OpenCV Haar Cascade
The Haar Cascade classifier bundled with OpenCV is the fastest option and requires zero additional installation. However it is a 2001-era algorithm with notably low accuracy — it struggles with faces at angles, partial occlusion, and varying lighting. It produces a high false-positive rate. It is used in this system only as a lightweight framing guide in the photo capture feed, not for recognition.

4.4 InsightFace (Selected)
InsightFace uses the buffalo_sc model which is based on RetinaFace for detection and ArcFace for recognition — both state-of-the-art architectures. It installs cleanly on Windows with a simple pip install insightface onnxruntime. The ONNX Runtime backend means there is no TensorFlow or PyTorch dependency. The model is downloaded automatically on first run (~200MB, one time only).

⦁	RetinaFace detection handles faces at angles, in crowds, and under varying lighting
⦁	ArcFace recognition generates highly discriminative 512-dimensional embeddings
⦁	Cosine similarity matching is fast since embeddings are pre-normalised
⦁	buffalo_sc is the lightweight variant — fast enough for real-time CPU inference

Note on buffalo_sc vs buffalo_l
buffalo_l is the full-size model with higher accuracy but slower speed. If you have a GPU or need higher accuracy in challenging conditions, change name='buffalo_sc' to name='buffalo_l' in app.py. The rest of the code is identical.


5. Installation & Setup
5.1 System Requirements
⦁	Windows 10/11, Ubuntu 20.04+, or macOS 12+
⦁	Python 3.8 – 3.11 recommended (3.12+ has some known issues with ONNX)
⦁	Webcam connected and not in use by another application
⦁	~500MB free disk space (for models and data)
⦁	Internet connection for first run (model download only)

5.2 Create Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

5.3 Install Dependencies
pip install flask flask-cors opencv-python numpy insightface onnxruntime pillow

5.4 Run the Application
python app.py

Then open your browser and go to http://localhost:5000

First Run
On the first launch, InsightFace will automatically download the buffalo_sc model files (~200MB). This only happens once. Subsequent startups are instant.


6. User Guide
Step 1 — Add Classes & Teachers
Navigate to the Classes & Teachers tab. Add your class sections (e.g. 10-A, CS-Sem3, Batch 2025) and teacher names. These populate the dropdown menus in the session setup.

Step 2 — Register Students
Navigate to the Students tab. For each student, enter:
⦁	Roll No. — unique identifier (e.g. 001, CS2301)
⦁	Full Name
⦁	Course — e.g. B.Tech, BCA, MCA
⦁	Year — 1st through 4th Year
⦁	Section — e.g. A, B, C

After adding a student, click the Photos button next to their name. A modal will open showing the live camera feed. Take 4–6 photos, ideally:
⦁	In the same lighting as the classroom
⦁	Facing slightly left, center, and slightly right
⦁	Without sunglasses or hats

Each captured photo appears as a thumbnail below the camera. You can delete any bad shot by hovering and clicking the × button.

Step 3 — Start a Session
Navigate to Live Session. Select a Teacher, enter the Subject name, and select the Class. Click Start Session. The camera feed will appear and recognition begins immediately.

The right panel shows a real-time attendance list. Students are automatically marked Present when their face is recognized. You can manually toggle any student using the green/grey switch.

Step 4 — Export CSV
Click Export CSV at any time during or after the session. A file is downloaded and also saved to data/attendance/ for your records. The CSV format is:

Attendance Report
Date,2025-02-25
Teacher,Dr. Smith
Subject,Mathematics
Class/Section,10-A

Roll No.,Name,Course,Year,Section,Status
001,Alice Johnson,B.Tech,2nd Year,A,Present
002,Bob Kumar,B.Tech,2nd Year,A,Absent
003,Carol Singh,B.Tech,2nd Year,B,Present

Total,3
Present,2
Absent,1


7. Challenges & How We Solved Them

Challenge 1: Browser Camera Permission (getUserMedia)
Problem: Modern browsers block camera access via JavaScript on non-HTTPS pages. Since the app runs on http://localhost, any attempt to use navigator.mediaDevices.getUserMedia() results in a security error.
Solution: The camera is accessed entirely server-side via OpenCV. The browser never asks for camera permission because it only ever makes HTTP requests to the Flask server, which returns MJPEG streams as <img> sources. This completely bypasses the browser security model.

Challenge 2: face_recognition & dlib Install Failures on Windows
Problem: The face_recognition library requires dlib which must be compiled from C++ source code. This requires cmake and Visual C++ Build Tools. Even when installed, version mismatches between dlib 20.x and face_recognition 1.3.0 cause runtime errors.
Solution: Replaced with InsightFace + ONNX Runtime. InsightFace distributes pre-compiled wheels and uses ONNX models that run directly via onnxruntime — no compilation step, no cmake, no C++ toolchain required.

Challenge 3: Real-time Performance on CPU
Problem: Running face recognition on every frame is too slow on a typical CPU, causing the video feed to stutter.
Solution: Recognition runs only every 8th frame (RECOG_EVERY = 8 in app.py). Bounding boxes and names from the last recognition pass are cached and reused for the frames in between. This gives smooth 20-30 FPS video with recognition updating ~3-4 times per second — fast enough for practical use.

Challenge 4: Same Student Marked Multiple Times
Problem: A student who stands in front of the camera for several seconds would trigger recognition on many frames and be logged as present many times.
Solution: The marked_today variable is a Python set. Sets only hold unique values, so adding the same roll number multiple times has no effect. Each student is recorded exactly once per session.

Challenge 5: File Organization
Problem: In early versions, the database.json, known_faces folder, and attendance_records folder were all dumped in the root of the project alongside source code, making the folder messy and hard to back up.
Solution: All runtime data is now organized under a single data/ directory with a clear hierarchy: data/db.json, data/students/<roll_no>/photos/, and data/attendance/. Photos use timestamped filenames to prevent collisions.

Challenge 6: DeepFace TensorFlow Conflict
Problem: DeepFace requires TensorFlow, which conflicts with other packages and does not install cleanly on Python 3.12+. Attempts to install it produced hundreds of dependency conflict errors.
Solution: Abandoned DeepFace entirely in favour of InsightFace which has no TensorFlow dependency. InsightFace uses ONNX format models which run via the lightweight onnxruntime package.


8. Frequently Asked Questions

Q: Can the system recognise multiple students simultaneously?
Yes. InsightFace detects and encodes all faces in a frame in a single pass. If three students walk through the door at the same time, all three are detected, identified, and marked in that same recognition cycle.
Q: What happens if a student is not registered in the system?
Unregistered faces are detected (a bounding box is drawn in blue/purple) but labelled Unknown. They are not marked in the attendance list. You can still manually mark any student present using the toggle switch in the right panel.
Q: How many photos per student are needed for good accuracy?
A minimum of 3 photos is needed, but 5-6 photos give significantly better results. Photos should be taken in the same lighting conditions as the classroom and from slightly varied angles. Avoid sunglasses, hats, or anything that partially covers the face.
Q: What does the recognition threshold of 0.4 mean?
The threshold is a cosine similarity score. A score of 1.0 means a perfect match, 0.0 means completely unrelated. At 0.4, the system accepts a match only if the live face is at least 40% similar to the stored embedding. Lower values are stricter (fewer false positives but more misses). Higher values are looser (more matches but risk of misidentification). You can adjust this in the identify() function in app.py.
Q: Does the system work with poor lighting?
InsightFace is more robust than older methods but still benefits from decent lighting. For best results, the room should be evenly lit without strong backlighting (e.g. avoid students standing in front of a bright window). Capturing training photos in the same lighting as the actual room dramatically improves accuracy.
Q: Can I use a different camera than the built-in webcam?
Yes. Change cv2.VideoCapture(0) to cv2.VideoCapture(1) or cv2.VideoCapture(2) in app.py. Index 0 is the default/built-in camera, higher indices are additional cameras. If you are unsure which index your camera is, try each one starting from 1.
Q: Does the system work without an internet connection?
Yes, after the first run. The first launch downloads the InsightFace buffalo_sc model (~200MB). Once downloaded, all models are cached locally and the system operates completely offline.
Q: Can the same session span multiple days?
No. Each session is a single event. The marked_today set resets to empty every time you click Start Session. For a new day, simply start a new session. All CSV exports are saved with the date in the filename so daily records remain separate.
Q: Where are attendance records saved?
All exported CSVs are saved to data/attendance/ inside the project folder. The filename format is attendance_<Class>_<Subject>_<Date>.csv. You can find and open these files at any time even without running the application.
Q: Can I run this on a GPU for better performance?
Yes. Change the providers parameter in app.py from ["CPUExecutionProvider"] to ["CUDAExecutionProvider"] if you have an NVIDIA GPU with CUDA installed. You will also need to install onnxruntime-gpu instead of onnxruntime. GPU inference is typically 5-10x faster and allows a lower RECOG_EVERY value for more frequent recognition updates.
Q: What is YOLO and should I use it instead of InsightFace?
YOLO (You Only Look Once) is a general object detection model. It can be trained or fine-tuned for face detection (YOLOv8-face). However, for this system InsightFace is the better choice because it handles both detection and recognition in one pipeline with a unified, well-tested API. YOLO would only replace the detection stage and you would still need a separate recognition model. InsightFace is simpler and more accurate for this specific use case.
Q: How do I improve recognition accuracy for students who wear glasses?
Capture some training photos of the student both with and without glasses. InsightFace handles glasses reasonably well, but having photos that represent the student's typical appearance in class will always give the best results.


9. Configuration Reference
These are the key constants in app.py that you can tune:

Variable	Default	Effect
RECOG_EVERY	8	Run recognition every N frames. Lower = more frequent but uses more CPU.
threshold in identify()	0.4	Cosine similarity threshold. Lower = stricter matching.
det_size in prepare()	(640,640)	Detection input resolution. Lower (e.g. 320,320) is faster but less accurate.
FRAME_WIDTH	1280	Camera capture width in pixels.
FRAME_HEIGHT	720	Camera capture height in pixels.
JPEG_QUALITY	80	Stream compression quality (0-100). Lower = faster but blurrier.


10. Troubleshooting

Symptom	Likely Cause	Fix
Camera feed is black	Another app is using the camera	Close other apps using webcam. Try VideoCapture(1).
Students showing as Unknown	Not enough training photos	Capture 5-6 more photos per student in room lighting.
High CPU usage	RECOG_EVERY too low	Increase RECOG_EVERY to 12 or 15 in app.py.
InsightFace models not downloading	No internet connection	Connect to internet for first run only.
Port 5000 already in use	Another app on port 5000	Change port in app.run() to 5001 or any free port.
Photos not saving	Camera index wrong	Check VideoCapture(0) vs VideoCapture(1) in app.py.
Face detected but wrong name	Threshold too high	Lower threshold from 0.4 to 0.35 in identify().
