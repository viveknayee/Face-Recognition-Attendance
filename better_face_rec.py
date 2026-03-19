# better_face_rec.py
import os
import cv2
import numpy as np
import face_recognition
import pickle
from math import atan2, degrees

# ----------------- CONFIG -----------------
IMAGES_DIR = "images"
ENCODINGS_FILE = "encodings_records.pickle"
DETECTOR = "hog"           # "hog" (fast) or "cnn" (more accurate if you have GPU)
MATCH_THRESHOLD = 0.5      # Euclidean distance threshold (tweak 0.5 - 0.65)
MIN_SEPARATION = 0.04      # require best vs second-best separation (helps avoid ambiguous matches)
MAX_IMAGE_DIM = 1000       # resize large images to speed up encoding
FACE_OUTPUT_SIZE = (256, 256)  # size used for aligned face before encoding
# -------------------------------------------


def adjust_gamma(image, gamma=1.1):
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_clahe_bgr(img_bgr):
    """Apply CLAHE to the Y channel of YCrCb (better contrast in low light)."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    ycrcb[:, :, 0] = y
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def preprocess_bgr(img_bgr):
    """Resize large images, CLAHE, gamma correction - returns BGR."""
    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    img_bgr = apply_clahe_bgr(img_bgr)
    img_bgr = adjust_gamma(img_bgr, gamma=1.08)
    return img_bgr


def align_face(face_rgb):
    """
    Align a cropped face image so eyes are horizontal.
    Input: face_rgb (RGB)
    Returns aligned RGB face (same size or rotated crop).
    """
    try:
        landmarks_list = face_recognition.face_landmarks(face_rgb)
        if not landmarks_list:
            return face_rgb
        lm = landmarks_list[0]
        if 'left_eye' not in lm or 'right_eye' not in lm:
            return face_rgb
        left = np.mean(lm['left_eye'], axis=0)
        right = np.mean(lm['right_eye'], axis=0)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        angle = degrees(atan2(dy, dx))
        # rotate around center
        h, w = face_rgb.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # negative to level eyes
        rotated = cv2.warpAffine(face_rgb, M, (w, h), flags=cv2.INTER_CUBIC)
        return rotated
    except Exception:
        return face_rgb


# ----------------- Record utilities -----------------
def save_records(records, path=ENCODINGS_FILE):
    """Records: list of dicts {path, mtime, name, encoding (numpy array)}"""
    with open(path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_records(path=ENCODINGS_FILE):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


# ----------------- Encoding builder (incremental + sync) -----------------
def build_or_update_records(images_dir=IMAGES_DIR, enc_file=ENCODINGS_FILE, detector=DETECTOR):
    """
    Load existing records and update (add new, remove deleted, re-encode modified).
    Returns list of records.
    """
    # Load old
    records = load_records(enc_file)
    rec_map = {r['path']: r for r in records}  # path -> record

    # Collect current image files
    current = {}
    for person in os.listdir(images_dir):
        pdir = os.path.join(images_dir, person)
        if not os.path.isdir(pdir):
            continue
        for fname in os.listdir(pdir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full = os.path.join(pdir, fname)
            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue
            current[full] = {'name': person, 'mtime': mtime}

    # Remove deleted files
    new_records = []
    for r in records:
        if r['path'] in current and abs(current[r['path']]['mtime'] - r['mtime']) < 1e-6:
            new_records.append(r)  # keep unchanged
    records = new_records
    rec_map = {r['path']: r for r in records}

    # Find new or modified files
    to_encode = []
    for path, info in current.items():
        if path not in rec_map or abs(rec_map[path]['mtime'] - info['mtime']) > 1e-6:
            to_encode.append((path, info['name'], info['mtime']))

    if to_encode:
        print(f"[INFO] Encoding {len(to_encode)} new/changed image(s)...")
    else:
        print("[INFO] No new images to encode.")

    for path, name, mtime in to_encode:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"[WARN] Can't open {path}, skipping.")
            continue

        img_bgr = preprocess_bgr(img_bgr)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Detect faces (fast HOG by default)
        boxes = face_recognition.face_locations(img_rgb, model=detector)
        if not boxes:
            print(f"[WARN] No face found in {os.path.basename(path)}, skipping.")
            continue

        # We'll encode each detected face (but store only first as that file is labeled)
        top, right, bottom, left = boxes[0]
        face_rgb = img_rgb[top:bottom, left:right]
        aligned = align_face(face_rgb)
        try:
            aligned_resized = cv2.resize(aligned, FACE_OUTPUT_SIZE)
        except Exception:
            aligned_resized = aligned

        encs = face_recognition.face_encodings(aligned_resized)
        if not encs:
            # fallback: try encoding from whole image
            encs = face_recognition.face_encodings(img_rgb, known_face_locations=[boxes[0]])
            if not encs:
                print(f"[WARN] Could not encode {os.path.basename(path)}, skipping.")
                continue

        rec = {
            'path': path,
            'mtime': mtime,
            'name': name,
            'encoding': encs[0]  # numpy array
        }
        records.append(rec)
        print(f"[INFO] Encoded: {os.path.basename(path)} -> {name}")

    # Save updated records
    save_records(records, enc_file)
    print(f"[INFO] Records saved. Total stored face entries: {len(records)}")
    return records


# ----------------- Build index for fast lookup -----------------
def build_index_from_records(records):
    if not records:
        return None, None, None
    encodings = np.array([r['encoding'] for r in records])
    names = [r['name'] for r in records]
    paths = [r['path'] for r in records]
    return encodings, names, paths


# ----------------- Recognition function -----------------
def recognize_frame(frame_bgr, encodings_index, names_index, detector=DETECTOR,
                    match_threshold=MATCH_THRESHOLD, min_separation=MIN_SEPARATION):
    """
    Input: BGR frame (full-size). Output: list of (top,right,bottom,left,name,dist)
    """
    results = []
    if encodings_index is None or len(encodings_index) == 0:
        return results

    # downscale for speed
    small = cv2.resize(frame_bgr, (0, 0), fx=0.2, fy=0.2)
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(small_rgb, model=detector)
    if not boxes:
        return results

    encs = face_recognition.face_encodings(small_rgb, boxes)
    # for each face found in frame
    for enc, (top, right, bottom, left) in zip(encs, boxes):
        # compute distances to all stored encodings
        dists = np.linalg.norm(encodings_index - enc, axis=1)  # Euclidean
        # aggregate min distance per person
        name_min = {}
        for i, nm in enumerate(names_index):
            d = float(dists[i])
            if nm not in name_min or d < name_min[nm]:
                name_min[nm] = d
        # find best match and second best
        sorted_names = sorted(name_min.items(), key=lambda x: x[1])
        if len(sorted_names) == 0:
            best_name, best_dist = "Unknown", None
            second_dist = None
        else:
            best_name, best_dist = sorted_names[0]
            second_dist = sorted_names[1][1] if len(sorted_names) > 1 else None

        # decide acceptance
        label = "Unknown"
        if best_dist is not None and best_dist < match_threshold:
            # optional separation check
            if second_dist is None or (second_dist - best_dist) >= min_separation:
                label = best_name.upper()
            else:
                # ambiguous: still allow if best is well below threshold
                if best_dist < (match_threshold * 0.85):
                    label = best_name.upper()

        # scale box coordinates back to original frame size
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
        results.append((top, right, bottom, left, label, best_dist))

    return results


# ----------------- CLI main loop -----------------
def main():
    print("[INFO] Syncing dataset and building encodings (incremental + align)...")
    records = build_or_update_records(IMAGES_DIR, ENCODINGS_FILE, detector=DETECTOR)
    enc_index, names_index, paths_index = build_index_from_records(records)

    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = recognize_frame(frame, enc_index, names_index, detector=DETECTOR,
                                  match_threshold=MATCH_THRESHOLD, min_separation=MIN_SEPARATION)

        for (top, right, bottom, left, name, dist) in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            if dist is None:
                label = f"{name}"
            else:
                confidence = round((1 - dist) * 100)
                label = f"{name} {confidence}%"
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Better Face Recognition - press q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # auto-detect if dataset changed on disk (simple check)
        # If you often add files while running, uncomment the block below to auto-update:
        # if os.path.exists(ENCODINGS_FILE):
        #     # quick heuristic: rebuild if encodings file older than any image
        #     enc_mtime = os.path.getmtime(ENCODINGS_FILE)
        #     newest_img = max((os.path.getmtime(p) for p in paths_index), default=0)
        #     if newest_img > enc_mtime:
        #         print("[INFO] Dataset changed - rebuilding encodings...")
        #         records = build_or_update_records(IMAGES_DIR, ENCODINGS_FILE, detector=DETECTOR)
        #         enc_index, names_index, paths_index = build_index_from_records(records)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
