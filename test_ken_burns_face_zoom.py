import cv2
import numpy as np
import pytest

import ken_burns_face_zoom as kb


def test_zoom_factor():
    assert kb.zoom_factor(0, 50) == 1
    assert kb.zoom_factor(1, 50) == pytest.approx(1.5)
    assert kb.zoom_factor(0, 50, 2.0) == pytest.approx(2.0)
    assert kb.zoom_factor(1, 50, 2.0) == pytest.approx(3.0)


def test_output_resolution_and_quality_options():
    args = kb.parse_args(["photos", "--resolution", "1080p", "--quality", "85"])
    assert (args.width, args.height) == (1920, 1080)
    assert args.quality == 85
    assert args.target == "chest"

    args = kb.parse_args(["photos", "--width", "640", "--height", "360"])
    assert (args.width, args.height) == (640, 360)
    assert kb.parse_args(["photos", "--start-target-size", "35"]).start_face_size == 35

    with pytest.raises(SystemExit):
        kb.parse_args(["photos", "--resolution", "720p", "--width", "640"])
    with pytest.raises(SystemExit):
        kb.parse_args(["photos", "--quality", "101"])
    with pytest.raises(SystemExit):
        kb.parse_args(["photos", "--width", "0"])


def test_build_video_applies_quality_option(monkeypatch):
    class Writer:
        def __init__(self):
            self.quality = None
            self.supports_quality = True

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.quality = (prop, value)
            return self.supports_quality

        def release(self):
            pass

    writer = Writer()
    monkeypatch.setattr(kb.cv2, "VideoWriter", lambda *args: writer)
    kb.build_video([], "out.mp4", [], quality=85)
    assert writer.quality == (cv2.VIDEOWRITER_PROP_QUALITY, 85)
    writer.supports_quality = False
    with pytest.raises(RuntimeError, match="does not support --quality"):
        kb.build_video([], "out.mp4", [], quality=85)


def test_initial_zoom_depends_on_face_size():
    out = (1280, 720)
    # solo portrait: face already fills >25% of frame -> no initial zoom
    assert kb.initial_zoom((4000, 3000), (1500, 800, 900, 1000), out, 25, 1.5) == 1.0
    # distant face in a group shot -> zoomed in so the face is ~25% of frame height
    z = kb.initial_zoom((4000, 3000), (1500, 800, 200, 200), out, 25, 1.5)
    assert z == pytest.approx(0.25 * 2250 / 200)
    assert kb.initial_zoom((4000, 3000), (1500, 800, 100, 100), out, 0, 1.5) == 1.0


def test_initial_zoom_capped_by_resolution():
    # 1600x900 source -> output 1280x720: at most 1.5x upscale => crop >= 853px wide
    z = kb.initial_zoom((1600, 900), (800, 400, 10, 10), (1280, 720), 25, 1.5)
    assert z == pytest.approx(1600 * 1.5 / 1280)
    assert kb.initial_zoom((1600, 900), (800, 400, 10, 10), (1280, 720), 25, 3) == pytest.approx(1600 * 3 / 1280)


def test_cluster_and_pick():
    a, b = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    thumb = np.zeros((4, 4, 3), np.uint8)
    clusters = kb.cluster_faces([(thumb, a, 1), (thumb, b, 1), (thumb, a * 2, 5)])
    assert [len(c["features"]) for c in clusters] == [2, 1]
    assert clusters[0]["size"] == 5
    feats = kb.choose_person(clusters, person=2)
    box = kb.pick_face([((0, 0, 5, 5), a), ((9, 9, 5, 5), b)], feats)
    assert box == (9, 9, 5, 5)
    assert kb.pick_face([((0, 0, 5, 5), a)], feats) is None
    with pytest.raises(SystemExit):
        kb.choose_person(clusters, person=3)


def test_crop_window_stays_inside_and_aspect():
    x, y, w, h = kb.crop_window(1000, 800, (990, 10), 2.0, 16 / 9)
    assert x >= 0 and y >= 0 and x + w <= 1000 and y + h <= 800
    assert w / h == pytest.approx(16 / 9, rel=0.02)


def test_find_images_orders(tmp_path):
    for n in ("b.jpg", "A.png", "c.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in kb.find_images(tmp_path, "name")] == ["A.png", "b.jpg"]
    assert len(kb.find_images(tmp_path, "random", seed=1)) == 2
    assert len(kb.find_images(tmp_path, "date")) == 2


def test_render_and_blur():
    img = np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    frames = list(kb.render_frames(img, (250, 150, 80, 80), 5, (320, 180), 40, 1.2, blur=3))
    assert len(frames) == 5 and frames[0].shape == (180, 320, 3)


def test_main_creates_video(tmp_path, monkeypatch):
    feat = np.array([1.0, 0.0])
    monkeypatch.setattr(kb, "detect_faces", lambda img, *a, **k: [((150, 100, 60, 60), feat)])
    img = np.full((300, 400, 3), 128, np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    out = tmp_path / "o.mp4"
    assert kb.main([str(tmp_path), "-o", str(out), "--seconds", "0.5", "--fps", "4",
                    "--width", "160", "--height", "90", "--person", "1",
                    "--people-sheet", str(tmp_path / "people.jpg"), "--target", "face"]) == 0
    assert out.stat().st_size > 0


def test_find_person_escalates_and_missing_excluded(tmp_path, monkeypatch):
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    calls = []

    def fake_detect(img, score=0.8, max_side=1280, flip=False):
        calls.append(flip)
        return [((1, 2, 3, 4), a)] if flip else [((5, 6, 7, 8), b)]

    monkeypatch.setattr(kb, "detect_faces", fake_detect)
    img = np.zeros((10, 10, 3), np.uint8)
    assert kb.find_person(img, [a]) == (1, 2, 3, 4)
    assert True in calls and calls[0] is False
    assert kb.find_person(img, [a], aggressive=False) is None

    cv2.imwrite(str(tmp_path / "a.png"), np.full((300, 400, 3), 128, np.uint8))
    out = tmp_path / "o.mp4"
    args = [str(tmp_path), "-o", str(out), "--seconds", "0.5", "--fps", "4", "--width", "160",
            "--height", "90", "--person", "1", "--people-sheet", str(tmp_path / "p.jpg"), "--target", "face"]
    monkeypatch.setattr(kb, "scan_people", lambda paths, n: [{"features": [a], "thumb": img, "size": 1}])
    monkeypatch.setattr(kb, "find_person", lambda *x, **k: None)
    kb.main(args)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0
    kb.main(args + ["--keep-missing"])
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0


def test_ctrl_c_saves_partial_video(tmp_path, monkeypatch, capsys):
    for n in ("a", "b", "c"):
        cv2.imwrite(str(tmp_path / f"{n}.png"), np.full((90, 160, 3), 100, np.uint8))
    paths = kb.find_images(tmp_path)
    calls = []

    def fake_find(img, features, aggressive):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return (60, 30, 30, 30)

    monkeypatch.setattr(kb, "find_person", fake_find)
    out = tmp_path / "o.mp4"
    used = kb.build_video(paths, out, [np.ones(2)], size=(160, 90), fps=4, seconds=1, target="face")
    assert used == 2
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 8
    err = capsys.readouterr().err
    assert "image 2 of 3" in err and "interrupted at image 3 of 3" in err


def test_chest_box_is_below_face_and_wider():
    x, y, w, h = kb.chest_box((100, 100, 50, 60))
    assert y + h / 2 == 100 + 2 * 60
    assert w > 50 and x + w / 2 == 125


def test_chest_target_prefers_blazepose_and_falls_back_to_face_estimate(monkeypatch):
    img = np.zeros((500, 500, 3), np.uint8)
    face = (100, 100, 50, 60)
    monkeypatch.setattr(kb, "pose_chest_box", lambda image, box: (80, 190, 160, 140))
    assert kb.target_box(img, face, "chest") == (80, 190, 160, 140)
    monkeypatch.setattr(kb, "pose_chest_box", lambda image, box: None)
    assert kb.target_box(img, face, "chest") == kb.chest_box(face)
    assert kb.target_box(img, face, "face") == face
