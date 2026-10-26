#!/usr/bin/env python3
"""Build a video from a folder of images, Ken Burns zooming on one person's face in each."""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


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


def detect_person_face(img):
    """Return (x, y, w, h) of the most prominent face, or None.

    The person of interest is taken to be the largest detected face.
    """
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    min_size = max(20, min(img.shape[:2]) // 40)
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_size, min_size))
        if len(faces):
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
    return None


def zoom_factor(t, zoom_percent, start_zoom_percent=0.0):
    """Zoom at progress t in [0, 1]; 1.0 means the full (aspect-fitted) frame."""
    return (1 + start_zoom_percent / 100.0) * (1 + zoom_percent / 100.0 * t)


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


def render_frames(img, face, n_frames, size, zoom_percent, start_zoom_percent=0.0, blur=0.0):
    """Yield n_frames BGR frames of the given size zooming towards the face."""
    out_w, out_h = size
    h, w = img.shape[:2]
    if face is None:
        face = (w // 2 - 1, h // 2 - 1, 2, 2)
    center = (face[0] + face[2] / 2, face[1] + face[3] / 2)
    src = prepare_image(img, face, blur)
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 1.0
        x0, y0, cw, ch = crop_window(w, h, center, zoom_factor(t, zoom_percent, start_zoom_percent),
                                     out_w / out_h)
        crop = src[y0:y0 + ch, x0:x0 + cw]
        yield cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def build_video(paths, output, size=(1280, 720), fps=30, seconds=3.0, zoom_percent=30.0,
                start_zoom_percent=0.0, blur=0.0):
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
            face = detect_person_face(img)
            if face is None:
                print(f"no face found in {path.name}; zooming on image centre", file=sys.stderr)
            for frame in render_frames(img, face, n_frames, size, zoom_percent,
                                       start_zoom_percent, blur):
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
    p.add_argument("--start-zoom", type=float, default=0.0,
                   help="percent zoom already applied at the start of each image (default 0)")
    p.add_argument("--blur", type=float, default=0.0,
                   help="blur the background; strength as percent of image size, e.g. 2 (default 0 = off)")
    p.add_argument("--seconds", type=float, default=3.0, help="seconds per image (default 3)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    args = p.parse_args(argv)
    if args.zoom < 0 or args.start_zoom < 0 or args.blur < 0:
        p.error("--zoom, --start-zoom and --blur must be >= 0")
    if args.seconds <= 0 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        p.error("--seconds, --fps, --width and --height must be > 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = find_images(args.folder, args.order, args.seed)
    if not paths:
        print("no images found", file=sys.stderr)
        return 1
    build_video(paths, args.output, (args.width, args.height), args.fps, args.seconds,
                args.zoom, args.start_zoom, args.blur)
    print(f"wrote {args.output} from {len(paths)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
