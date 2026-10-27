#!/usr/bin/env python3
"""Build a video from a folder of images, Ken Burns zooming on one person's chest in each."""
import argparse
import hashlib
import random
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime

MODEL_DIR = Path(__file__).resolve().parent / "models"
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "detector": ("face_detection_yunet_2023mar.onnx", _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
                 "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "recognizer": ("face_recognition_sface_2021dec.onnx", _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                   "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
    "pose": ("pose_landmarker_full.task",
             "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task",
             "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1"),
}
MATCH_THRESHOLD = 0.363  # SFace cosine similarity threshold for "same person"
DETECT_MAX_SIDE = 1280
RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
}


def load_image(path):
    """Load an image as BGR uint8, honouring EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def capture_date(path):
    """EXIF capture date string, falling back to file modification time."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(ExifTags.IFD.Exif)
            for tag in EXIF_DATE_TAGS:
                value = sub.get(tag) or exif.get(tag)
                if value:
                    return str(value)
    except Exception:
        pass
    import datetime
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y:%m:%d %H:%M:%S")


def find_images(folder, order="name", seed=None):
    paths = [p for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if order == "name":
        paths.sort(key=lambda p: p.name.lower())
    elif order == "date":
        paths.sort(key=lambda p: (capture_date(p), p.name.lower()))
    elif order == "random":
        paths.sort(key=lambda p: p.name.lower())
        random.Random(seed).shuffle(paths)
    else:
        raise ValueError(f"unknown order: {order}")
    return paths


def model_path(key):
    """Path of a model file, downloaded (and checksum-verified) on first use."""
    name, url, sha256 = MODELS[key]
    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        print(f"downloading {name} ...", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError(f"checksum mismatch for {name}")
        path.write_bytes(data)
    return str(path)


_engine = {}
_pose_engine = None
_POSE_NOT_SUPPLIED = object()


def detect_faces(img, score=0.8, max_side=DETECT_MAX_SIDE, flip=False, include_landmarks=False):
    """Return a list of (box, feature) for every face; box is (x, y, w, h) in pixels
    and feature is an identity embedding comparable with `similarity`.

    score is the detector confidence threshold, max_side the longest side the image
    is detected at, and flip mirrors the image first (boxes are mapped back)."""
    if not _engine:
        _engine["det"] = cv2.FaceDetectorYN.create(model_path("detector"), "", (320, 320), 0.8)
        _engine["rec"] = cv2.FaceRecognizerSF.create(model_path("recognizer"), "")
    det, rec = _engine["det"], _engine["rec"]
    det.setScoreThreshold(score)
    if flip:
        img = cv2.flip(img, 1)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1 else img
    det.setInputSize((small.shape[1], small.shape[0]))
    _, rows = det.detect(small)
    faces = []
    for row in (rows if rows is not None else []):
        row = row.copy()
        row[:14] /= scale
        feature = rec.feature(rec.alignCrop(img, row))[0]
        x, y, fw, fh = (int(round(v)) for v in row[:4])
        x, y = max(0, x), max(0, y)
        fw, fh = min(fw, w - x), min(fh, h - y)
        if fw > 1 and fh > 1:
            landmarks = row[4:14].reshape(5, 2).copy()
            if flip:
                x = w - x - fw
                landmarks[:, 0] = w - landmarks[:, 0]
            face = ((x, y, fw, fh), feature)
            faces.append(face + (landmarks,) if include_landmarks else face)
    return faces


def similarity(a, b):
    """Cosine similarity of two face embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cluster_faces(faces, threshold=MATCH_THRESHOLD):
    """Group (thumbnail, feature) entries by identity.

    Returns clusters, most frequent first; each is {"thumb", "features", "size"}
    where thumb is the largest face seen for that person."""
    clusters = []
    for thumb, feature, size in faces:
        best, best_sim = None, threshold
        for c in clusters:
            sim = max(similarity(feature, f) for f in c["features"])
            if sim >= best_sim:
                best, best_sim = c, sim
        if best is None:
            best = {"thumb": thumb, "features": [], "size": 0}
            clusters.append(best)
        best["features"].append(feature)
        if size > best["size"]:
            best["thumb"], best["size"] = thumb, size
    clusters.sort(key=lambda c: -len(c["features"]))
    return clusters


def scan_people(paths, count):
    """Detect and cluster the people in the first `count` images."""
    faces = []
    for i, path in enumerate(paths[:count], 1):
        print(f"scanning image {i} of {min(count, len(paths))}: {path.name}", file=sys.stderr)
        try:
            img = load_image(path)
        except Exception as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        for (x, y, w, h), feature in detect_faces(img):
            pad = int(0.3 * max(w, h))
            thumb = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            faces.append((thumb, feature, w * h))
    return cluster_faces(faces)


def people_sheet(clusters, thumb_size=200, max_people=20):
    """Contact sheet image with a numbered thumbnail for each person."""
    cells = []
    for i, c in enumerate(clusters[:max_people], 1):
        cell = np.full((thumb_size + 30, thumb_size, 3), 255, np.uint8)
        t = c["thumb"]
        s = thumb_size / max(t.shape[:2])
        t = cv2.resize(t, (max(1, int(t.shape[1] * s)), max(1, int(t.shape[0] * s))))
        cell[:t.shape[0], :t.shape[1]] = t
        cv2.putText(cell, f"#{i} ({len(c['features'])} photos)", (4, thumb_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(cell)
    cols = min(5, len(cells))
    rows = [np.hstack(cells[i:i + cols] + [np.full_like(cells[0], 255)] * (cols - len(cells[i:i + cols])))
            for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def choose_person(clusters, person=None, sheet_path="people.jpg"):
    """Pick one cluster: from `person` (1-based) or by asking the user."""
    if not clusters:
        raise SystemExit("no faces found in the scanned images")
    shown = min(len(clusters), 20)
    if person is None:
        cv2.imwrite(sheet_path, people_sheet(clusters))
        print(f"found {len(clusters)} people; see {sheet_path} (numbered thumbnails)")
        if not sys.stdin.isatty():
            raise SystemExit(f"choose with --person N (1-{shown}) when not running interactively")
        while person is None:
            answer = input(f"Which person should the video focus on? [1-{shown}]: ").strip()
            if answer.isdigit() and 1 <= int(answer) <= shown:
                person = int(answer)
    if not 1 <= person <= shown:
        raise SystemExit(f"--person must be between 1 and {shown}")
    return clusters[person - 1]["features"]


def pick_face_entry(faces, features, threshold=MATCH_THRESHOLD):
    """Return the matched face entry, preserving optional landmark data."""
    best, best_sim = None, threshold
    for face in faces:
        box, feature = face[:2]
        sim = max(similarity(feature, f) for f in features)
        if sim >= best_sim:
            best, best_sim = face, sim
    return best


def pick_face(faces, features, threshold=MATCH_THRESHOLD):
    """Return the box of the face best matching the chosen person, or None."""
    face = pick_face_entry(faces, features, threshold)
    return face[0] if face is not None else None


def chest_box(face):
    """Estimate the upper-chest region from a face box.

    This is used only when BlazePose cannot find visible shoulders."""
    x, y, fw, fh = face
    w, h = 2.4 * fw, 1.8 * fh
    return (int(round(x + fw / 2 - w / 2)), int(round(y + 2.0 * fh - h / 2)), int(round(w)), int(round(h)))


def pose_engine():
    """Create the MediaPipe BlazePose detector on first use."""
    global _pose_engine
    if _pose_engine is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is required for --target chest; run 'pip install -r requirements.txt'"
            ) from exc
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path("pose")),
            num_poses=1,
            min_pose_detection_confidence=0.4,
            output_segmentation_masks=True,
        )
        _pose_engine = (mp, vision.PoseLandmarker.create_from_options(options))
    return _pose_engine


def face_yaw(landmarks):
    """Estimate left/right head yaw from YuNet's nose and eye landmarks."""
    if landmarks is None:
        return 0.0
    eye_center = (landmarks[0] + landmarks[1]) / 2
    eye_distance = np.linalg.norm(landmarks[0] - landmarks[1])
    return float(np.clip((landmarks[2][0] - eye_center[0]) / (eye_distance + 1e-6), -1.0, 1.0))


def pose_matches_face(face, face_landmarks, pose_nose):
    """Whether a pose is anchored to the selected face rather than a nearby person."""
    if face_landmarks is None:
        return True
    x, y, fw, fh = face
    margin = max(fw, fh) * 0.35
    in_face = x - margin <= pose_nose[0] <= x + fw + margin and y - margin <= pose_nose[1] <= y + fh + margin
    close_to_face_nose = np.linalg.norm(pose_nose - face_landmarks[2]) <= max(fw, fh) * 0.75
    return in_face and close_to_face_nose


def pose_detection(img, face, landmarks=None):
    """Return the face-anchored BlazePose result and its source crop.

    The face-centred crop keeps a different person in a group photo from being
    selected by Pose. Its generous lower extent includes the shoulders and,
    when visible, hips used to scale the torso and outline."""
    x, y, fw, fh = face
    img_h, img_w = img.shape[:2]
    crop_size = min(max(2, int(round(max(5 * fw, 8 * fh)))), img_w, img_h)
    center_x = x + fw / 2 + face_yaw(landmarks) * fw
    center_y = y + 3 * fh
    x0 = int(round(min(max(center_x - crop_size / 2, 0), img_w - crop_size)))
    y0 = int(round(min(max(center_y - crop_size / 2, 0), img_h - crop_size)))
    x1, y1 = x0 + crop_size, y0 + crop_size
    if crop_size < 2:
        return None

    crop = img[y0:y1, x0:x1]
    mp, detector = pose_engine()
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if result.pose_landmarks and landmarks is not None:
        pose_nose = np.array((x0 + result.pose_landmarks[0][0].x * crop_size,
                              y0 + result.pose_landmarks[0][0].y * crop_size))
        if not pose_matches_face(face, landmarks, pose_nose):
            return None
    return result, (x0, y0, x1, y1)


def pose_chest_box(img, face, detection=_POSE_NOT_SUPPLIED, full_chest=False):
    """Return a BlazePose-derived chest box for `face`, or None if unavailable."""
    if detection is _POSE_NOT_SUPPLIED:
        detection = pose_detection(img, face)
    if detection is None:
        return None
    _, _, fw, fh = face
    result, (x0, y0, x1, y1) = detection
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    left_shoulder, right_shoulder = landmarks[11], landmarks[12]
    if min(left_shoulder.visibility, right_shoulder.visibility) < 0.5:
        return None

    def point(landmark):
        return np.array((x0 + landmark.x * (x1 - x0), y0 + landmark.y * (y1 - y0)))

    shoulders = (point(left_shoulder) + point(right_shoulder)) / 2
    shoulder_width = float(np.linalg.norm(point(left_shoulder) - point(right_shoulder)))
    left_hip, right_hip = landmarks[23], landmarks[24]
    if min(left_hip.visibility, right_hip.visibility) >= 0.5:
        hips = (point(left_hip) + point(right_hip)) / 2
        torso = float(np.linalg.norm(hips - shoulders))
    else:
        torso = shoulder_width
    if shoulder_width < 2 or torso < 2:
        return None

    if full_chest:
        width = max(shoulder_width * 1.6, fw * 2.4)
        if 'hips' in locals():
            height = max(torso * 1.2, shoulder_width * 1.2)
            center = shoulders + (hips - shoulders) * 0.5
        else:
            height = max(shoulder_width * 1.6, fh * 2.2)
            center = shoulders + np.array((0, height * 0.35))
    else:
        width = max(shoulder_width * 1.25, fw * 2)
        height = max(torso * 0.9, shoulder_width)
        center = shoulders + (hips - shoulders) * 0.4 if 'hips' in locals() else shoulders + np.array((0, torso * 0.35))
    return (int(round(center[0] - width / 2)), int(round(center[1] - height / 2)),
            int(round(width)), int(round(height)))


def pose_subject_mask(img, detection):
    """Return BlazePose's soft person outline mapped to the source image."""
    if detection is None:
        return None
    result, (x0, y0, x1, y1) = detection
    if not result.segmentation_masks:
        return None
    mask = result.segmentation_masks[0].numpy_view()
    mask = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    full_mask = np.zeros(img.shape[:2], np.float32)
    full_mask[y0:y1, x0:x1] = np.clip(mask, 0.0, 1.0)
    return cv2.GaussianBlur(full_mask, (0, 0), max(img.shape[:2]) * 0.003)


def target_box(img, face, target, detection=None, full_chest=False):
    """Return the requested target box, estimating chest position from the face."""
    if target != "chest":
        return face
    return pose_chest_box(img, face, detection, full_chest) or chest_box(face)


def zoom_factor(t, zoom_percent, start_zoom=1.0):
    """Zoom at progress t in [0, 1]; 1.0 means the full (aspect-fitted) frame.

    start_zoom is the zoom at t=0; zoom_percent is then added on top of it."""
    return start_zoom * (1 + zoom_percent / 100.0 * t)


def animation_progress(t, settle_at=0.8):
    """Return eased pan and zoom progress, with a final zoom-only settle."""
    def ease_out(value):
        return (1 - np.exp(-5 * value)) / (1 - np.exp(-5))

    if t < settle_at:
        return ease_out(t / settle_at), 0.9 * ease_out(t / settle_at)
    return 1.0, 0.9 + 0.1 * ease_out((t - settle_at) / (1 - settle_at))


def initial_zoom(img_size, target, out_size, target_fraction, max_upscale):
    """Zoom needed at the start so the target is `target_fraction` (percent) of the
    frame height, never below 1 and never so deep that the crop has fewer than
    out_width / max_upscale source pixels across."""
    img_w, img_h = img_size
    out_w, out_h = out_size
    aspect = out_w / out_h
    base_w = min(img_w, img_h * aspect)
    base_h = base_w / aspect
    zoom = max(1.0, target_fraction / 100.0 * base_h / target[3]) if target_fraction > 0 else 1.0
    return max(1.0, min(zoom, base_w * max_upscale / out_w))


def full_chest_zoom(img_size, chest, out_size):
    """Return the closest aspect-fitted crop that contains the complete chest."""
    img_w, img_h = img_size
    out_w, out_h = out_size
    aspect = out_w / out_h
    base_w = min(img_w, img_h * aspect)
    base_h = base_w / aspect
    return max(1.0, min(base_w / chest[2], base_h / chest[3]))


def crop_window_f(img_w, img_h, center, zoom, aspect):
    """Like crop_window but returns float (x0, y0, w, h) for sub-pixel rendering."""
    base_w = min(img_w, img_h * aspect)
    base_h = base_w / aspect
    w = max(2.0, base_w / zoom)
    h = max(2.0, base_h / zoom)
    x0 = min(max(center[0] - w / 2, 0), img_w - w)
    y0 = min(max(center[1] - h / 2, 0), img_h - h)
    return x0, y0, w, h


def crop_window(img_w, img_h, center, zoom, aspect):
    """Return integer (x0, y0, w, h) of a window with the given aspect ratio,
    centred on `center` as far as possible while staying inside the image."""
    x0, y0, w, h = crop_window_f(img_w, img_h, center, zoom, aspect)
    return int(round(x0)), int(round(y0)), max(2, int(round(w))), max(2, int(round(h)))


def fallback_subject_mask(shape, face, feather=0.6):
    """Return the prior face-based approximation when pose segmentation fails."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    x, y, fw, fh = face
    center = (int(x + fw / 2), int(y + fh * 1.6))
    axes = (max(1, int(fw * 1.2)), max(1, int(fh * 2.6)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    k = int(max(fw, fh) * feather) | 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def prepare_image(img, face, blur, mask=None):
    """Return the image to animate (optionally with the background blurred)."""
    if not blur:
        return img
    k = (int(max(img.shape[:2]) * blur / 100.0) | 1)
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    mask = (mask if mask is not None else fallback_subject_mask(img.shape, face))[..., None]
    return (img * mask + blurred * (1 - mask)).astype(np.uint8)


def render_frames(img, target, n_frames, size, zoom_percent, start_zoom=1.0, blur=0.0, upsample=True,
                  blur_face=None, blur_mask=None):
    """Yield n_frames BGR frames of the given size zooming towards the target.

    The crop window moves linearly from its start to its end position, so pan and
    zoom finish together. Frames are rendered with sub-pixel precision (no integer
    rounding of the crop) to avoid jerky motion; `upsample` XX"""
    out_w, out_h = size
    h, w = img.shape[:2]
    if target is None:
        target = (w // 2 - 1, h // 2 - 1, 2, 2)
    center = (target[0] + target[2] / 2, target[1] + target[3] / 2)
    src = prepare_image(img, blur_face or target, blur, blur_mask)
    aspect = out_w / out_h
    start = crop_window_f(w, h, center, zoom_factor(0.0, zoom_percent, start_zoom), aspect)
    end = crop_window_f(w, h, center, zoom_factor(1.0, zoom_percent, start_zoom), aspect)
    # Crop to the area that will ever be shown, so scaling never touches the rest.
    ux0 = max(0, int(min(start[0], end[0])) - 2)
    uy0 = max(0, int(min(start[1], end[1])) - 2)
    ux1 = min(w, int(np.ceil(max(start[0] + start[2], end[0] + end[2]))) + 2)
    uy1 = min(h, int(np.ceil(max(start[1] + start[3], end[1] + end[3]))) + 2)
    src = src[uy0:uy1, ux0:ux1]
    magnification = out_w / end[2]  # output pixels per source pixel at the end
    if magnification >= 1.0:
        f = 1.5 * magnification if upsample else 1.0
    else:
        f = out_w / start[2]  # source is larger than the output: shrink once, no aliasing
    if f != 1.0:
        src = cv2.resize(src, None, fx=f, fy=f,
                         interpolation=cv2.INTER_AREA if f < 1.0 else cv2.INTER_CUBIC)
    start_center = (start[0] + start[2] / 2, start[1] + start[3] / 2)
    end_center = (end[0] + end[2] / 2, end[1] + end[3] / 2)
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 1.0
        pan_t, zoom_t = animation_progress(t)
        cw, ch = ((1 - zoom_t) * a + zoom_t * b for a, b in zip(start[2:], end[2:]))
        center = tuple((1 - pan_t) * a + pan_t * b for a, b in zip(start_center, end_center))
        x0 = min(max(center[0] - cw / 2, 0), w - cw)
        y0 = min(max(center[1] - ch / 2, 0), h - ch)
        sx, sy = out_w / (cw * f), out_h / (ch * f)
        m = np.float64([[sx, 0, -(x0 - ux0) * f * sx], [0, sy, -(y0 - uy0) * f * sy]])
        yield cv2.warpAffine(src, m, (out_w, out_h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)


# Progressively more aggressive passes used to find a known person:
# (detector score, detection size, mirrored, match threshold)
SEARCH_PASSES = [
    (0.8, DETECT_MAX_SIDE, False, MATCH_THRESHOLD),
    (0.5, DETECT_MAX_SIDE, False, MATCH_THRESHOLD - 0.05),
    (0.5, 2 * DETECT_MAX_SIDE, False, MATCH_THRESHOLD - 0.05),
    (0.4, DETECT_MAX_SIDE, True, MATCH_THRESHOLD - 0.08),
    (0.3, 2 * DETECT_MAX_SIDE, True, MATCH_THRESHOLD - 0.08),
]


def find_person(img, features, aggressive=True, include_landmarks=False):
    """Locate the chosen person's face box in img, or None.

    Since we know whose face we want, a weak match is accepted when stricter
    searches fail: lower detector confidence, larger detection size (small faces),
    mirrored image (profiles facing the other way) and a relaxed match threshold."""
    for score, max_side, flip, threshold in SEARCH_PASSES[:None if aggressive else 1]:
        kwargs = {"include_landmarks": True} if include_landmarks else {}
        face = pick_face_entry(
            detect_faces(img, score, max_side, flip, **kwargs), features, threshold
        )
        if face is not None:
            return face if include_landmarks else face[0]
    return None


def build_video(paths, output, features, size=(1280, 720), fps=30, seconds=3.0, zoom_percent=30.0,
                start_face_size=25.0, max_upscale=1.5, blur=0.0, keep_missing=False,
                aggressive=True, upsample=True, quality=None, target="chest", full_chest=False):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {output}")
    if quality is not None and not writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality):
        writer.release()
        raise RuntimeError("video writer backend does not support --quality")
    n_frames = max(1, int(round(seconds * fps)))
    total = len(paths)
    used = 0
    try:
        for i, path in enumerate(paths, 1):
            print(f"image {i} of {total}: {path.name}", file=sys.stderr)
            try:
                img = load_image(path)
            except Exception as exc:
                print(f"skipping {path}: {exc}", file=sys.stderr)
                continue
            matched = find_person(img, features, aggressive, include_landmarks=True)
            if matched is None:
                face, face_landmarks = None, None
            elif isinstance(matched[0], tuple):
                face = matched[0]
                face_landmarks = matched[2] if len(matched) > 2 else None
            else:
                face, face_landmarks = matched, None
            blur_face = face
            pose = pose_detection(img, face, face_landmarks) if face is not None and (target == "chest" or blur) else None
            chest = pose_chest_box(img, face, pose) if target == "chest" and face is not None else None
            full_box = pose_chest_box(img, face, pose, full_chest=True) if full_chest and chest is not None else None
            full_chest_active = full_box is not None and full_chest_zoom(
                (img.shape[1], img.shape[0]), full_box, size
            ) > 1.0
            estimated_chest = chest_box(face) if target == "chest" and face is not None and chest is None else None
            box = full_box if full_chest_active else (chest or estimated_chest or face)
            if box is None:
                if not keep_missing:
                    print(f"chosen person not found in {path.name}; excluding it", file=sys.stderr)
                    continue
                print(f"chosen person not found in {path.name}; zooming on image centre", file=sys.stderr)
                start_zoom = 1.0
                image_zoom = zoom_percent
            elif full_chest_active:
                start_zoom = 1.0
                image_zoom = (full_chest_zoom((img.shape[1], img.shape[0]), box, size) - 1.0) * 100
            else:
                start_zoom = initial_zoom((img.shape[1], img.shape[0]), box, size,
                                          start_face_size, max_upscale)
                image_zoom = zoom_percent
            target_name = (
                "chest" if chest is not None
                else ("chest-estimate" if estimated_chest is not None else ("face" if face is not None else "centre"))
            )
            print(
                f"target={target_name}; full_chest={'yes' if full_chest_active else 'no'}; "
                f"initial_zoom={start_zoom * 100:.0f}%; "
                f"final_zoom={start_zoom * (1 + image_zoom / 100) * 100:.0f}%",
                file=sys.stderr,
            )
            mask = pose_subject_mask(img, pose) if blur else None
            for frame in render_frames(img, box, n_frames, size, image_zoom, start_zoom, blur, upsample,
                                       blur_face, mask):
                writer.write(frame)
            used += 1
    except KeyboardInterrupt:
        print(f"\ninterrupted at image {i} of {total}; saving the video made so far", file=sys.stderr)
    finally:
        writer.release()
    return used


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", help="folder containing the images")
    p.add_argument("-o", "--output", default="output.mp4")
    p.add_argument("--order", choices=["random", "date", "name"], default="name",
                   help="image order: random, date captured, or filename (default: name)")
    p.add_argument("--seed", type=int, help="random seed for --order random")
    p.add_argument("--zoom", type=float, default=30.0, help="percent zoom during each image (default 30)")
    p.add_argument("--start-target-size", "--start-face-size", dest="start_face_size", type=float, default=25.0,
                   help="initial target height as percent of frame height; distant targets start "
                        "zoomed in to this size, larger targets are left alone (default 25, 0 = off)")
    p.add_argument("--max-upscale", type=float, default=1.5,
                   help="max initial zoom, as a multiple of the output video's pixel resolution (default 1.5)")
    p.add_argument("--target", choices=["face", "chest"], default="chest",
                   help="what to zoom on: the chest detected with BlazePose (default), or the face")
    p.add_argument("--full-chest", action="store_true",
                   help="when chest landmarks are reliable, zoom from the full image to the complete chest area")
    p.add_argument("--keep-missing", action="store_true",
                   help="keep images where the chosen person isn't found (zoom on the centre) "
                        "instead of excluding them")
    p.add_argument("--no-aggressive", action="store_true",
                   help="only use the strict face search (faster, but misses turned-away/small faces)")
    p.add_argument("--scan", type=int, default=5,
                   help="number of images to scan for people to choose from (default 5)")
    p.add_argument("--person", type=int, help="number of the person to follow (skips the prompt)")
    p.add_argument("--people-sheet", default="people.jpg",
                   help="where to write the numbered thumbnails of people found (default people.jpg)")
    p.add_argument("--no-upsample", action="store_true",
                   help="do not pre-enlarge the cropped area (faster, slightly less smooth)")
    p.add_argument("--blur", type=float, default=0.0,
                   help="blur the background; strength as percent of image size, e.g. 2 (default 0 = off)")
    p.add_argument("--seconds", type=float, default=3.0, help="seconds per image (default 3)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--resolution", choices=RESOLUTIONS,
                   help="output resolution preset: 480p, 720p, 1080p, 1440p or 2160p")
    p.add_argument("--width", type=int, help="custom output width (default 1280)")
    p.add_argument("--height", type=int, help="custom output height (default 720)")
    p.add_argument("--quality", type=int,
                   help="encoder quality from 0 (lowest) to 100 (highest); backend-dependent")
    args = p.parse_args(argv)
    if args.resolution and (args.width is not None or args.height is not None):
        p.error("--resolution cannot be combined with --width or --height")
    if args.resolution:
        args.width, args.height = RESOLUTIONS[args.resolution]
    else:
        args.width = 1280 if args.width is None else args.width
        args.height = 720 if args.height is None else args.height
    if args.zoom < 0 or args.start_face_size < 0 or args.blur < 0:
        p.error("--zoom, --start-face-size and --blur must be >= 0")
    if args.max_upscale < 1 or args.scan < 1:
        p.error("--max-upscale must be >= 1 and --scan >= 1")
    if args.seconds <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        p.error("--seconds, --fps, --width and --height must be > 0")
    if args.quality is not None and not 0 <= args.quality <= 100:
        p.error("--quality must be between 0 and 100")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = find_images(args.folder, args.order, args.seed)
    if not paths:
        print("no images found", file=sys.stderr)
        return 1
    features = choose_person(scan_people(paths, args.scan), args.person, args.people_sheet)
    used = build_video(paths, args.output, features, (args.width, args.height), args.fps, args.seconds,
                args.zoom, args.start_face_size, args.max_upscale, args.blur,
                args.keep_missing, not args.no_aggressive, not args.no_upsample, args.quality, args.target,
                args.full_chest)
    print(f"wrote {args.output} from {used} of {len(paths)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
