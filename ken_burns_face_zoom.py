#!/usr/bin/env python3
"""Build a video from a folder of images, Ken Burns zooming on one person's face in each."""
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
}
MATCH_THRESHOLD = 0.363  # SFace cosine similarity threshold for "same person"
DETECT_MAX_SIDE = 1280


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


def detect_faces(img):
    """Return a list of (box, feature) for every face; box is (x, y, w, h) in pixels
    and feature is an identity embedding comparable with `similarity`."""
    if not _engine:
        _engine["det"] = cv2.FaceDetectorYN.create(model_path("detector"), "", (320, 320), 0.8)
        _engine["rec"] = cv2.FaceRecognizerSF.create(model_path("recognizer"), "")
    det, rec = _engine["det"], _engine["rec"]
    h, w = img.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
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
            faces.append(((x, y, fw, fh), feature))
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
    for path in paths[:count]:
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


def pick_face(faces, features, threshold=MATCH_THRESHOLD):
    """Return the box of the face best matching the chosen person, or None."""
    best, best_sim = None, threshold
    for box, feature in faces:
        sim = max(similarity(feature, f) for f in features)
        if sim >= best_sim:
            best, best_sim = box, sim
    return best


def zoom_factor(t, zoom_percent, start_zoom=1.0):
    """Zoom at progress t in [0, 1]; 1.0 means the full (aspect-fitted) frame.

    start_zoom is the zoom at t=0; zoom_percent is then added on top of it."""
    return start_zoom * (1 + zoom_percent / 100.0 * t)


def initial_zoom(img_size, face, out_size, face_fraction, max_upscale):
    """Zoom needed at the start so the face is `face_fraction` (percent) of the
    frame height, never below 1 and never so deep that the crop has fewer than
    out_width / max_upscale source pixels across."""
    img_w, img_h = img_size
    out_w, out_h = out_size
    aspect = out_w / out_h
    base_w = min(img_w, img_h * aspect)
    base_h = base_w / aspect
    zoom = max(1.0, face_fraction / 100.0 * base_h / face[3]) if face_fraction > 0 else 1.0
    return max(1.0, min(zoom, base_w * max_upscale / out_w))


def crop_window(img_w, img_h, center, zoom, aspect):
    """Return integer (x0, y0, w, h) of a window with the given aspect ratio,
    centred on `center` as far as possible while staying inside the image."""
    base_w = min(img_w, img_h * aspect)
    base_h = base_w / aspect
    w = max(2.0, base_w / zoom)
    h = max(2.0, base_h / zoom)
    x0 = min(max(center[0] - w / 2, 0), img_w - w)
    y0 = min(max(center[1] - h / 2, 0), img_h - h)
    return int(round(x0)), int(round(y0)), max(2, int(round(w))), max(2, int(round(h)))


def subject_mask(shape, face, feather=0.6):
    """Float mask (1 = subject) as a soft ellipse around the face and body."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    x, y, fw, fh = face
    center = (int(x + fw / 2), int(y + fh * 1.6))
    axes = (max(1, int(fw * 1.2)), max(1, int(fh * 2.6)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    k = int(max(fw, fh) * feather) | 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def prepare_image(img, face, blur):
    """Return the image to animate (optionally with the background blurred)."""
    if not blur:
        return img
    k = (int(max(img.shape[:2]) * blur / 100.0) | 1)
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    mask = subject_mask(img.shape, face)[..., None]
    return (img * mask + blurred * (1 - mask)).astype(np.uint8)


def render_frames(img, face, n_frames, size, zoom_percent, start_zoom=1.0, blur=0.0):
    """Yield n_frames BGR frames of the given size zooming towards the face."""
    out_w, out_h = size
    h, w = img.shape[:2]
    if face is None:
        face = (w // 2 - 1, h // 2 - 1, 2, 2)
    center = (face[0] + face[2] / 2, face[1] + face[3] / 2)
    src = prepare_image(img, face, blur)
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 1.0
        x0, y0, cw, ch = crop_window(w, h, center, zoom_factor(t, zoom_percent, start_zoom),
                                     out_w / out_h)
        crop = src[y0:y0 + ch, x0:x0 + cw]
        yield cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def build_video(paths, output, features, size=(1280, 720), fps=30, seconds=3.0, zoom_percent=30.0,
                start_face_size=25.0, max_upscale=1.5, blur=0.0):
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {output}")
    n_frames = max(1, int(round(seconds * fps)))
    try:
        for path in paths:
            try:
                img = load_image(path)
            except Exception as exc:
                print(f"skipping {path}: {exc}", file=sys.stderr)
                continue
            face = pick_face(detect_faces(img), features)
            if face is None:
                print(f"chosen person not found in {path.name}; zooming on image centre", file=sys.stderr)
                start_zoom = 1.0
            else:
                start_zoom = initial_zoom((img.shape[1], img.shape[0]), face, size,
                                          start_face_size, max_upscale)
            for frame in render_frames(img, face, n_frames, size, zoom_percent, start_zoom, blur):
                writer.write(frame)
    finally:
        writer.release()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", help="folder containing the images")
    p.add_argument("-o", "--output", default="output.mp4")
    p.add_argument("--order", choices=["random", "date", "name"], default="name",
                   help="image order: random, date captured, or filename (default: name)")
    p.add_argument("--seed", type=int, help="random seed for --order random")
    p.add_argument("--zoom", type=float, default=30.0, help="percent zoom during each image (default 30)")
    p.add_argument("--start-face-size", type=float, default=25.0,
                   help="initial face height as percent of frame height; distant faces start "
                        "zoomed in to this size, larger faces are left alone (default 25, 0 = off)")
    p.add_argument("--max-upscale", type=float, default=1.5,
                   help="max initial zoom, as a multiple of the output video's pixel resolution (default 1.5)")
    p.add_argument("--scan", type=int, default=5,
                   help="number of images to scan for people to choose from (default 5)")
    p.add_argument("--person", type=int, help="number of the person to follow (skips the prompt)")
    p.add_argument("--people-sheet", default="people.jpg",
                   help="where to write the numbered thumbnails of people found (default people.jpg)")
    p.add_argument("--blur", type=float, default=0.0,
                   help="blur the background; strength as percent of image size, e.g. 2 (default 0 = off)")
    p.add_argument("--seconds", type=float, default=3.0, help="seconds per image (default 3)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args(argv)
    if args.zoom < 0 or args.start_face_size < 0 or args.blur < 0:
        p.error("--zoom, --start-face-size and --blur must be >= 0")
    if args.max_upscale < 1 or args.scan < 1:
        p.error("--max-upscale must be >= 1 and --scan >= 1")
    if args.seconds <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        p.error("--seconds, --fps, --width and --height must be > 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = find_images(args.folder, args.order, args.seed)
    if not paths:
        print("no images found", file=sys.stderr)
        return 1
    features = choose_person(scan_people(paths, args.scan), args.person, args.people_sheet)
    build_video(paths, args.output, features, (args.width, args.height), args.fps, args.seconds,
                args.zoom, args.start_face_size, args.max_upscale, args.blur)
    print(f"wrote {args.output} from {len(paths)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
