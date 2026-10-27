import cv2
import numpy as np
import pytest
from types import SimpleNamespace
from pathlib import Path

import ken_burns_face_zoom as kb


def test_zoom_factor():
    assert kb.zoom_factor(0, 50) == 1
    assert kb.zoom_factor(1, 50) == pytest.approx(1.5)
    assert kb.zoom_factor(0, 50, 2.0) == pytest.approx(2.0)
    assert kb.zoom_factor(1, 50, 2.0) == pytest.approx(3.0)


def test_animation_progress_settles_pan_before_the_final_zoom():
    assert kb.animation_progress(0) == pytest.approx((0, 0))
    pan, zoom = kb.animation_progress(0.8)
    assert pan == 1
    assert zoom == pytest.approx(0.9)
    pan, zoom = kb.animation_progress(0.9)
    assert pan == 1
    assert 0.9 < zoom < 1
    assert kb.animation_progress(1) == pytest.approx((1, 1))


def test_output_resolution_and_quality_options():
    args = kb.parse_args(["photos", "--resolution", "1080p", "--quality", "85"])
    assert (args.width, args.height) == (1920, 1080)
    assert args.quality == 85
    assert args.target == "chest"
    assert not args.full_chest
    assert args.chest_hold == 0.5

    args = kb.parse_args(["photos", "--width", "640", "--height", "360"])
    assert (args.width, args.height) == (640, 360)
    assert kb.parse_args(["photos", "--start-target-size", "35"]).start_face_size == 35
    assert kb.parse_args(["photos", "--full-chest"]).full_chest
    assert kb.parse_args(["photos", "--chest-hold", "1.25"]).chest_hold == 1.25

    with pytest.raises(SystemExit):
        kb.parse_args(["photos", "--resolution", "720p", "--width", "640"])
    with pytest.raises(SystemExit):
        kb.parse_args(["photos", "--quality", "101"])
    with pytest.raises(SystemExit):
        kb.parse_args(["photos", "--width", "0"])
    with pytest.raises(SystemExit):
        kb.parse_args(["photos", "--chest-hold", "-1"])


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
    assert z == pytest.approx(0.25 * 720 / (200 * (720 / 3000)))
    assert kb.initial_zoom((4000, 3000), (1500, 800, 100, 100), out, 0, 1.5) == 1.0


def test_initial_zoom_capped_by_resolution():
    # 1600x900 source -> output 1280x720: at most 1.5x upscale => crop >= 853px wide
    z = kb.initial_zoom((1600, 900), (800, 400, 10, 10), (1280, 720), 25, 1.5)
    assert z == pytest.approx(1600 * 1.5 / 1280)
    assert kb.initial_zoom((1600, 900), (800, 400, 10, 10), (1280, 720), 25, 3) == pytest.approx(1600 * 3 / 1280)


def test_full_chest_zoom_fits_the_complete_target_without_an_upscale_cap():
    chest = (1000, 700, 300, 500)
    assert kb.full_chest_zoom((4000, 3000), chest, (1280, 720)) == pytest.approx((720 / 500) / (720 / 3000))


def test_torso_box_spans_from_head_to_above_knees():
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]
    for index, x, y in ((0, 0.5, 0.15), (11, 0.35, 0.3), (12, 0.65, 0.3),
                        (23, 0.4, 0.6), (24, 0.6, 0.6), (25, 0.4, 0.9), (26, 0.6, 0.9)):
        landmarks[index] = SimpleNamespace(x=x, y=y, visibility=1.0)
    result = SimpleNamespace(pose_landmarks=[landmarks])
    box = kb.torso_box((200, 100, 50, 60), (result, (0, 0, 500, 500)))
    assert box[1] < 100
    assert box[1] + box[3] < 450
    assert box[3] > 300


def test_pose_must_be_anchored_to_the_selected_face():
    face = (100, 100, 60, 60)
    landmarks = np.array(((110, 115), (150, 115), (130, 130), (115, 145), (145, 145)))
    assert kb.pose_matches_face(face, landmarks, np.array((130, 130)))
    assert not kb.pose_matches_face(face, landmarks, np.array((220, 130)))


TEST_IMAGES = Path(__file__).with_name("test_images")


@pytest.mark.skipif(not (TEST_IMAGES / "IMG382.jpeg").exists(),
                    reason="requires the LFS test fixture")
def test_img382_left_person_does_not_use_the_other_persons_chest():
    image = kb.load_image(TEST_IMAGES / "IMG382.jpeg")
    face, _, landmarks = min(kb.detect_faces(image, include_landmarks=True),
                             key=lambda candidate: candidate[0][0])

    assert face[0] + face[2] / 2 < image.shape[1] / 2
    pose = kb.pose_detection(image, face, landmarks)
    assert kb.pose_chest_box(image, face, pose) is None
    assert kb.target_box(image, face, "chest", pose) == kb.chest_box(face)


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


def test_opening_window_keeps_head_visible_when_torso_does_not_fit():
    torso = (300, 100, 300, 1200)
    window = kb.opening_window(1000, 2000, torso, 1, 16 / 9)
    assert window[1] <= torso[1] <= window[1] + window[3]
    assert not kb.target_fits_window(torso, window)


def test_find_images_orders(tmp_path):
    for n in ("b.jpg", "A.png", "c.txt"):
        (tmp_path / n).write_bytes(b"x")
    assert [p.name for p in kb.find_images(tmp_path, "name")] == ["A.png", "b.jpg"]
    assert len(kb.find_images(tmp_path, "random", seed=1)) == 2
    assert len(kb.find_images(tmp_path, "date")) == 2


def test_render_and_blur():
    img = np.random.default_rng(0).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    mask = np.zeros(img.shape[:2], np.float32)
    mask[100:300, 200:400] = 1
    frames = list(kb.render_frames(img, (250, 150, 80, 80), 5, (320, 180), 40, 1.2, blur=3,
                                   blur_mask=mask))
    assert len(frames) == 5 and frames[0].shape == (180, 320, 3)


def test_blur_mask_always_preserves_the_matched_face():
    img = np.random.default_rng(1).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    face = (40, 30, 30, 20)
    result = kb.prepare_image(img, face, blur=10, mask=np.zeros(img.shape[:2], np.float32))
    assert np.array_equal(result[30:50, 40:70], img[30:50, 40:70])


def test_scale_to_fit_zoom_one_letterboxes_without_cropping():
    img = np.full((200, 100, 3), 255, np.uint8)
    frame = kb.render_view(img, (160, 90), 1.0, (50, 100))
    assert frame.shape == (90, 160, 3)
    assert np.all(frame[:, :55] == 0)
    assert np.all(frame[:, 60:100] == 255)


def test_target_visibility_requires_the_complete_chest_box():
    img = np.zeros((100, 100, 3), np.uint8)
    assert kb.target_is_visible((10, 10, 80, 80), img.shape)
    assert not kb.target_is_visible((10, 50, 80, 80), img.shape)


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

    def fake_find(img, features, aggressive, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return (60, 30, 30, 30)

    monkeypatch.setattr(kb, "find_person", fake_find)
    out = tmp_path / "o.mp4"
    used = kb.build_video(paths, out, [np.ones(2)], size=(160, 90), fps=4, seconds=1, target="face")
    assert used == 2
    cap = cv2.VideoCapture(str(out))
    # Each source gets its 250 ms initial hold; the second also crossfades from the first.
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 11
    err = capsys.readouterr().err
    assert "image 2 of 3" in err and "interrupted at image 3 of 3" in err


def test_chest_box_is_below_face_and_wider():
    x, y, w, h = kb.chest_box((100, 100, 50, 60))
    assert y + h / 2 == 100 + 2 * 60
    assert w > 50 and x + w / 2 == 125


def test_chest_target_prefers_blazepose_and_falls_back_to_face_estimate(monkeypatch):
    img = np.zeros((500, 500, 3), np.uint8)
    face = (100, 100, 50, 60)
    monkeypatch.setattr(
        kb, "pose_chest_box", lambda image, box, detection=None, full_chest=False: (80, 190, 160, 140)
    )
    assert kb.target_box(img, face, "chest") == (80, 190, 160, 140)
    monkeypatch.setattr(kb, "pose_chest_box", lambda image, box, detection=None, full_chest=False: None)
    assert kb.target_box(img, face, "chest") == kb.chest_box(face)
    assert kb.target_box(img, face, "face") == face


def test_build_video_estimates_chest_and_keeps_zoom_when_shoulders_are_missing(tmp_path, monkeypatch):
    class Writer:
        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    face = (100, 100, 50, 60)
    calls = []
    image_path = tmp_path / "source.png"
    cv2.imwrite(str(image_path), np.zeros((300, 400, 3), np.uint8))
    monkeypatch.setattr(kb.cv2, "VideoWriter", lambda *args: Writer())
    monkeypatch.setattr(kb, "find_person", lambda *args, **kwargs: (face, np.ones(2), np.zeros((5, 2))))
    monkeypatch.setattr(kb, "pose_detection", lambda *args: None)

    def fake_render(image, target, n_frames, size, zoom_percent, *args):
        calls.append((target, zoom_percent))
        yield np.zeros((size[1], size[0], 3), np.uint8)

    monkeypatch.setattr(kb, "render_frames", fake_render)
    kb.build_video([image_path], tmp_path / "out.mp4", [], size=(160, 90), fps=1, seconds=1,
                   zoom_percent=37, target="chest")
    assert calls == [(kb.chest_box(face), 37)]


def test_pose_chest_box_uses_face_width_as_minimum():
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
    landmarks[11] = SimpleNamespace(x=0.3, y=0.3, visibility=1.0)
    landmarks[12] = SimpleNamespace(x=0.7, y=0.3, visibility=1.0)
    landmarks[23] = SimpleNamespace(x=0.4, y=0.8, visibility=1.0)
    landmarks[24] = SimpleNamespace(x=0.6, y=0.8, visibility=1.0)
    result = SimpleNamespace(pose_landmarks=[landmarks])
    box = kb.pose_chest_box(np.zeros((500, 500, 3), np.uint8), (100, 100, 50, 60),
                            (result, (0, 0, 500, 500)))
    assert box is not None
    assert box[2] >= 100
    full_box = kb.pose_chest_box(np.zeros((500, 500, 3), np.uint8), (100, 100, 50, 60),
                                 (result, (0, 0, 500, 500)), full_chest=True)
    assert full_box[2] > box[2] and full_box[3] > box[3]

    landmarks[23].visibility = landmarks[24].visibility = 0.0
    shoulder_full_box = kb.pose_chest_box(np.zeros((500, 500, 3), np.uint8), (100, 100, 50, 60),
                                          (result, (0, 0, 500, 500)), full_chest=True)
    assert shoulder_full_box[2] > box[2] and shoulder_full_box[3] > box[3]
