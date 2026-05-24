import os
import cv2
import json
import csv
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from flask_cors import CORS
from insightface.app import FaceAnalysis

# ── InsightFace ────────────────────────────────────────────────────────────────
face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

app = Flask(__name__)
CORS(app)

# ── Directory layout ───────────────────────────────────────────────────────────
#
#  attendance_app/
#  ├── app.py
#  ├── requirements.txt
#  ├── templates/index.html
#  └── data/
#      ├── db.json
#      ├── students/<roll_no>/photos/*.jpg
#      ├── attendance/*.csv
#      └── unknowns/<session_id>/*.jpg   ← NEW: one folder per session
#
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DB_PATH      = os.path.join(DATA_DIR, "db.json")
STUDENTS_DIR = os.path.join(DATA_DIR, "students")
ATT_DIR      = os.path.join(DATA_DIR, "attendance")
UNKNOWNS_DIR = os.path.join(DATA_DIR, "unknowns")

for d in [DATA_DIR, STUDENTS_DIR, ATT_DIR, UNKNOWNS_DIR]:
    os.makedirs(d, exist_ok=True)

def photos_dir(roll_no):
    return os.path.join(STUDENTS_DIR, str(roll_no), "photos")

# ── Database ───────────────────────────────────────────────────────────────────
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            return json.load(f)
    return {"students": [], "teachers": [], "classes": []}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def student_photo_count(roll_no):
    d = photos_dir(roll_no)
    return len(os.listdir(d)) if os.path.exists(d) else 0

# ── Embeddings ─────────────────────────────────────────────────────────────────
known_embeddings = []
known_roll_nos   = []

def load_all_embeddings():
    global known_embeddings, known_roll_nos
    known_embeddings, known_roll_nos = [], []
    if not os.path.exists(STUDENTS_DIR):
        return
    for roll_no in os.listdir(STUDENTS_DIR):
        pd = photos_dir(roll_no)
        if not os.path.isdir(pd):
            continue
        for fname in sorted(os.listdir(pd)):
            img = cv2.imread(os.path.join(pd, fname))
            if img is None:
                continue
            faces = face_app.get(img)
            if faces:
                known_embeddings.append(faces[0].normed_embedding)
                known_roll_nos.append(roll_no)
    print(f"[Embeddings] {len(known_embeddings)} embeddings "
          f"for {len(set(known_roll_nos))} students.")

def add_embedding(roll_no, img):
    faces = face_app.get(img)
    if faces:
        known_embeddings.append(faces[0].normed_embedding)
        known_roll_nos.append(str(roll_no))
        return True
    return False

def remove_embeddings(roll_no):
    global known_embeddings, known_roll_nos
    pairs = [(e, r) for e, r in zip(known_embeddings, known_roll_nos)
             if r != str(roll_no)]
    known_embeddings = [p[0] for p in pairs]
    known_roll_nos   = [p[1] for p in pairs]

def identify(embedding, threshold=0.4):
    if not known_embeddings:
        return None, 0.0
    scores     = np.dot(known_embeddings, embedding)
    best_idx   = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score >= threshold:
        return known_roll_nos[best_idx], best_score
    return None, best_score

# ── Camera ─────────────────────────────────────────────────────────────────────
camera      = None
camera_lock = threading.Lock()

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def release_camera():
    global camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            camera = None

def grab_frame():
    if current_session.get("active"):
        with camera_lock:
            ret, frame = get_camera().read()
        return frame if ret else None
    else:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

# ── Session state ──────────────────────────────────────────────────────────────
current_session  = {}
marked_today     = set()       # roll numbers marked present
unknown_detected = {}          # track unknowns: face_hash -> {count, saved, first_seen}
unknown_photos   = []          # list of saved unknown photo filenames this session
frame_counter    = 0
cached_faces     = []
RECOG_EVERY      = 8

# How many consecutive recognition cycles an unknown must appear before saving.
# RECOG_EVERY=8 means one cycle ≈ 8 frames.  3 cycles ≈ ~24 frames (~0.8s at 30fps)
# — fast enough to catch a brief intruder, slow enough to skip single-frame noise.
UNKNOWN_SAVE_THRESHOLD = 3

# Cosine-similarity threshold to treat two unknown embeddings as the "same" person
UNKNOWN_MERGE_THRESHOLD = 0.55

# ── Unknown helpers ────────────────────────────────────────────────────────────
def session_unknowns_dir():
    """Folder for unknowns captured in the current session."""
    sid = current_session.get("session_id", "default")
    d   = os.path.join(UNKNOWNS_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d

def find_or_create_unknown_slot(embedding):
    """
    Match this embedding against already-seen unknown embeddings using cosine
    similarity.  Returns the slot key (string) — either an existing one if the
    face matches, or a new key if it's a fresh unknown.

    This replaces the old `embedding_hash` approach which was broken:
      int(np.argmax(emb)*1000 + np.sum(emb*100))
    that expression is dominated by np.sum which varies only in the last decimal
    place across embeddings, so almost every face hashed to the same bucket and
    the counter never accumulated properly.
    """
    best_key   = None
    best_score = -1.0

    for key, slot in unknown_detected.items():
        score = float(np.dot(slot["embedding"], embedding))
        if score > best_score:
            best_score = score
            best_key   = key

    if best_key is not None and best_score >= UNKNOWN_MERGE_THRESHOLD:
        return best_key

    # New unique unknown — use a simple incrementing key
    new_key = f"unk_{len(unknown_detected)}"
    unknown_detected[new_key] = {
        "embedding":  embedding,
        "count":      0,
        "saved":      False,
        "first_seen": datetime.now().strftime("%H:%M:%S"),
    }
    return new_key

def maybe_save_unknown(embedding, frame, bbox):
    """
    Track how many recognition cycles this unknown face has appeared.
    Once UNKNOWN_SAVE_THRESHOLD cycles are hit, crop and save one photo.
    Returns True if a new photo was just saved.
    """
    key = find_or_create_unknown_slot(embedding)
    unknown_detected[key]["count"] += 1

    if (not unknown_detected[key]["saved"]
            and unknown_detected[key]["count"] >= UNKNOWN_SAVE_THRESHOLD):
        unknown_detected[key]["saved"] = True

        # Crop face from frame with some padding
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad  = 30
        h, w = frame.shape[:2]
        x1c  = max(0, x1 - pad)
        y1c  = max(0, y1 - pad)
        x2c  = min(w, x2 + pad)
        y2c  = min(h, y2 + pad)
        crop = frame[y1c:y2c, x1c:x2c]

        fname = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        fpath = os.path.join(session_unknowns_dir(), fname)
        cv2.imwrite(fpath, crop)
        unknown_photos.append(fname)
        print(f"[Unknown] Saved photo: {fname}  (slot={key}, "
              f"seen={unknown_detected[key]['count']} cycles)")
        return True
    return False

# ── Frame rendering ────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, name, roll_no, is_marked):
    if is_marked:
        color = (0, 255, 120)      # green  — present
    elif roll_no:
        color = (0, 200, 255)      # cyan   — recognised but not yet marked
    else:
        color = (60, 60, 230)      # red-ish — unknown

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    c = 16
    for sx, sy, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame, (sx,sy), (sx+dx*c, sy), color, 3)
        cv2.line(frame, (sx,sy), (sx, sy+dy*c), color, 3)

    label = f"{name}  {roll_no}" if roll_no and name != "Unknown" else name
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+10, y1), color, -1)
    cv2.putText(frame, label, (x1+5, y1-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

def process_frame(frame):
    db, results = load_db(), []
    for face in face_app.get(frame):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        roll_no, name   = None, "Unknown"

        if known_embeddings:
            roll_no, _ = identify(face.normed_embedding)
            if roll_no:
                s = next((s for s in db["students"]
                          if str(s["roll_no"]) == roll_no), None)
                if s:
                    name = s["name"]
                    if roll_no not in marked_today:
                        marked_today.add(roll_no)
                        print(f"[Marked] {name} ({roll_no})")

        # If still unknown, track and possibly save photo
        if roll_no is None:
            maybe_save_unknown(face.normed_embedding, frame, face.bbox)

        is_marked = (roll_no in marked_today) if roll_no else False
        results.append((x1, y1, x2, y2, roll_no, name, is_marked))
    return results

def generate_frames():
    global frame_counter, cached_faces
    while current_session.get("active"):
        with camera_lock:
            ret, frame = get_camera().read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % RECOG_EVERY == 0:
            cached_faces = process_frame(frame)

        for args in cached_faces:
            draw_box(frame, *args)

        # HUD bar
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (frame.shape[1], 50), (8,8,12), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

        # Session time display
        start_dt  = current_session.get("_start_dt")
        elapsed   = ""
        if start_dt:
            secs    = int((datetime.now() - start_dt).total_seconds())
            elapsed = f"  |  {secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"

        hud = (f"  {current_session.get('subject','')}  |  "
               f"{current_session.get('class_name','')}  |  "
               f"{current_session.get('teacher','')}  |  "
               f"Present: {len(marked_today)}  |  "
               f"Unknowns: {len(unknown_photos)}"
               f"{elapsed}")
        cv2.putText(frame, hud, (8, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0,255,150), 1, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

def generate_capture_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                cv2.equalizeHist(gray), 1.1, 5, minSize=(60,60))
            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,229,160),2)
                c=14
                for sx,sy,dx,dy in [(x,y,1,1),(x+w,y,-1,1),
                                    (x,y+h,1,-1),(x+w,y+h,-1,-1)]:
                    cv2.line(frame,(sx,sy),(sx+dx*c,sy),(0,229,160),2)
                    cv2.line(frame,(sx,sy),(sx,sy+dy*c),(0,229,160),2)
            _, buf = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
    finally:
        cap.release()

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

# ── Students ───────────────────────────────────────────────────────────────────
@app.route("/api/students", methods=["GET"])
def get_students():
    db = load_db()
    return jsonify([{**s, "photos": student_photo_count(s["roll_no"])}
                    for s in db["students"]])

@app.route("/api/students", methods=["POST"])
def add_student():
    db, data = load_db(), request.json
    if any(str(s["roll_no"]) == str(data["roll_no"]) for s in db["students"]):
        return jsonify({"error": "Roll number already exists"}), 400
    student = {
        "roll_no": str(data["roll_no"]),
        "name":    data["name"].strip(),
        "course":  data.get("course","").strip(),
        "year":    data.get("year","").strip(),
        "section": data.get("section","").strip(),
    }
    db["students"].append(student)
    save_db(db)
    os.makedirs(photos_dir(student["roll_no"]), exist_ok=True)
    return jsonify(student), 201

@app.route("/api/students/<roll_no>", methods=["DELETE"])
def delete_student(roll_no):
    import shutil
    db = load_db()
    db["students"] = [s for s in db["students"]
                      if str(s["roll_no"]) != roll_no]
    save_db(db)
    student_root = os.path.join(STUDENTS_DIR, roll_no)
    if os.path.exists(student_root):
        shutil.rmtree(student_root)
    remove_embeddings(roll_no)
    return jsonify({"ok": True})

# ── Photos ─────────────────────────────────────────────────────────────────────
@app.route("/api/students/<roll_no>/capture", methods=["POST"])
def capture_photo(roll_no):
    frame = grab_frame()
    if frame is None:
        return jsonify({"ok": False, "error": "Camera unavailable"}), 500
    pd = photos_dir(roll_no)
    os.makedirs(pd, exist_ok=True)
    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    cv2.imwrite(os.path.join(pd, fname), frame)
    face_found = add_embedding(roll_no, frame)
    count = len(os.listdir(pd))
    return jsonify({"ok": True, "photos": count,
                    "face_found": face_found, "filename": fname})

@app.route("/api/students/<roll_no>/photos", methods=["GET"])
def list_photos(roll_no):
    pd = photos_dir(roll_no)
    return jsonify(sorted(os.listdir(pd)) if os.path.exists(pd) else [])

@app.route("/api/students/<roll_no>/photos/<filename>", methods=["GET"])
def serve_photo(roll_no, filename):
    fpath = os.path.join(photos_dir(roll_no), filename)
    return send_file(fpath, mimetype="image/jpeg") if os.path.exists(fpath) else ("", 404)

@app.route("/api/students/<roll_no>/photos/<filename>", methods=["DELETE"])
def delete_photo(roll_no, filename):
    fpath = os.path.join(photos_dir(roll_no), filename)
    if os.path.exists(fpath):
        os.remove(fpath)
    remove_embeddings(roll_no)
    pd = photos_dir(roll_no)
    if os.path.exists(pd):
        for f in os.listdir(pd):
            img = cv2.imread(os.path.join(pd, f))
            if img is not None:
                add_embedding(roll_no, img)
    count = len(os.listdir(pd)) if os.path.exists(pd) else 0
    return jsonify({"ok": True, "photos": count})

# ── Unknown photos ─────────────────────────────────────────────────────────────
@app.route("/api/unknowns/<filename>", methods=["GET"])
def serve_unknown(filename):
    sid   = current_session.get("session_id", "default")
    fpath = os.path.join(UNKNOWNS_DIR, sid, filename)
    return send_file(fpath, mimetype="image/jpeg") if os.path.exists(fpath) else ("", 404)

# ── Classes ────────────────────────────────────────────────────────────────────
@app.route("/api/classes", methods=["GET"])
def get_classes():
    return jsonify(load_db().get("classes", []))

@app.route("/api/classes", methods=["POST"])
def add_class():
    db  = load_db()
    cls = request.json["name"].strip()
    if cls not in db.get("classes", []):
        db.setdefault("classes", []).append(cls)
        save_db(db)
    return jsonify({"ok": True})

@app.route("/api/classes/<cls>", methods=["DELETE"])
def delete_class(cls):
    db = load_db()
    db["classes"] = [c for c in db.get("classes",[]) if c != cls]
    save_db(db)
    return jsonify({"ok": True})

# ── Teachers ───────────────────────────────────────────────────────────────────
@app.route("/api/teachers", methods=["GET"])
def get_teachers():
    return jsonify(load_db().get("teachers", []))

@app.route("/api/teachers", methods=["POST"])
def add_teacher():
    db = load_db()
    t  = request.json["name"].strip()
    if t not in db.get("teachers",[]):
        db.setdefault("teachers",[]).append(t)
        save_db(db)
    return jsonify({"ok": True})

@app.route("/api/teachers/<path:name>", methods=["DELETE"])
def delete_teacher(name):
    db = load_db()
    db["teachers"] = [t for t in db.get("teachers",[]) if t != name]
    save_db(db)
    return jsonify({"ok": True})

# ── Session ────────────────────────────────────────────────────────────────────
@app.route("/api/session/start", methods=["POST"])
def start_session():
    global current_session, marked_today, frame_counter
    global cached_faces, unknown_detected, unknown_photos

    data     = request.json
    now      = datetime.now()
    # Unique session ID used to name the unknowns folder
    sess_id  = now.strftime("%Y%m%d_%H%M%S")

    current_session = {
        "active":     True,
        "teacher":    data.get("teacher",""),
        "subject":    data.get("subject",""),
        "class_name": data.get("class_name",""),
        "date":       now.strftime("%Y-%m-%d"),
        "start_time": now.strftime("%H:%M:%S"),   # ← lecture start time
        "end_time":   None,                        # ← filled on stop
        "session_id": sess_id,
        "_start_dt":  now,                         # datetime object for elapsed timer
    }
    marked_today     = set()
    frame_counter    = 0
    cached_faces     = []
    unknown_detected = {}
    unknown_photos   = []

    load_all_embeddings()
    return jsonify({"ok": True, "session": _safe_session()})

@app.route("/api/session/stop", methods=["POST"])
def stop_session():
    global current_session
    current_session["active"]   = False
    current_session["end_time"] = datetime.now().strftime("%H:%M:%S")
    release_camera()
    return jsonify({
        "ok":             True,
        "unknown_count":  len(unknown_photos),
        "unknown_photos": unknown_photos,
    })

@app.route("/api/session/status", methods=["GET"])
def session_status():
    return jsonify({
        "active":         current_session.get("active", False),
        "session":        _safe_session(),
        "marked":         list(marked_today),
        "unknown_count":  len(unknown_photos),
        "unknown_photos": unknown_photos,
    })

@app.route("/api/session/mark", methods=["POST"])
def manual_mark():
    roll_no = str(request.json["roll_no"])
    if roll_no in marked_today:
        marked_today.discard(roll_no)
        return jsonify({"status": "unmarked"})
    marked_today.add(roll_no)
    return jsonify({"status": "marked"})

def _safe_session():
    """Return session dict without the internal _start_dt datetime object."""
    return {k: v for k, v in current_session.items() if k != "_start_dt"}

# ── Feeds ──────────────────────────────────────────────────────────────────────
@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture_feed")
def capture_feed():
    return Response(generate_capture_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Attendance ─────────────────────────────────────────────────────────────────
@app.route("/api/attendance/preview", methods=["GET"])
def preview_attendance():
    db       = load_db()
    students = sorted(db["students"], key=lambda s: s["roll_no"])
    rows = [{
        "roll_no": s["roll_no"], "name": s["name"],
        "course":  s.get("course",""), "year": s.get("year",""),
        "section": s.get("section",""),
        "status":  "Present" if s["roll_no"] in marked_today else "Absent"
    } for s in students]

    # Append unknown rows
    for i, fname in enumerate(unknown_photos, start=1):
        rows.append({
            "roll_no": f"UNK-{i:02d}",
            "name":    "Unknown Person",
            "course":  "—", "year": "—", "section": "—",
            "status":  "Intruder",
            "photo":   fname,
        })
    return jsonify(rows)

@app.route("/api/attendance/export", methods=["POST"])
def export_attendance():
    db       = load_db()
    students = sorted(db["students"], key=lambda s: s["roll_no"])

    date_str   = current_session.get("date",       datetime.now().strftime("%Y-%m-%d"))
    start_time = current_session.get("start_time", "—")
    end_time   = current_session.get("end_time",   "—")
    teacher    = current_session.get("teacher",    "Unknown")
    subject    = current_session.get("subject",    "Unknown")
    class_name = current_session.get("class_name", "Unknown")

    filename = (f"attendance_{class_name}_{subject}_{date_str}"
                f"_{start_time.replace(':','-')}.csv").replace(" ","_")
    filepath = os.path.join(ATT_DIR, filename)

    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        # Header block
        w.writerow(["Attendance Report"])
        w.writerow(["Date",          date_str])
        w.writerow(["Lecture Start", start_time])       # ← NEW
        w.writerow(["Lecture End",   end_time])          # ← NEW
        w.writerow(["Teacher",       teacher])
        w.writerow(["Subject",       subject])
        w.writerow(["Class/Section", class_name])
        w.writerow([])

        # Student rows
        w.writerow(["Roll No.", "Name", "Course", "Year", "Section", "Status"])
        for s in students:
            w.writerow([
                s["roll_no"], s["name"],
                s.get("course",""), s.get("year",""), s.get("section",""),
                "Present" if s["roll_no"] in marked_today else "Absent"
            ])

        # Unknown rows  ← NEW
        if unknown_photos:
            w.writerow([])
            w.writerow(["UNKNOWN / UNREGISTERED INDIVIDUALS DETECTED"])
            w.writerow(["#", "Identifier", "Status", "Photo File"])
            for i, fname in enumerate(unknown_photos, start=1):
                w.writerow([i, f"Unknown-{i:02d}", "Intruder", fname])

        # Summary
        w.writerow([])
        total   = len(students)
        present = sum(1 for s in students if s["roll_no"] in marked_today)
        w.writerow(["Total Students", total])
        w.writerow(["Present",        present])
        w.writerow(["Absent",         total - present])
        w.writerow(["Unknown Persons Detected", len(unknown_photos)])   # ← NEW

    return send_file(filepath, as_attachment=True,
                     download_name=filename, mimetype="text/csv")

# ── Boot ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  ╔════════════════════════════════════════╗")
    print("  ║  Attendance System  —  Starting...     ║")
    print("  ╚════════════════════════════════════════╝")
    print("  NOTE: First run downloads InsightFace models (~200MB). One time only.")
    print("\n  Open →  http://localhost:5000\n")
    load_all_embeddings()
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
