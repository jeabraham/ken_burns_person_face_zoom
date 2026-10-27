# ken_burns_person_face_zoom

Builds a video from a folder of images. The first N images are scanned for
people; you pick which person to follow (from numbered thumbnails in
`people.jpg`, or with `--person K`). In every image that person's face is
found, then MediaPipe BlazePose detects that person's shoulders and torso to
place a Ken Burns zoom on their chest. Face detection and recognition use
OpenCV's YuNet and SFace models; chest detection uses the MediaPipe
PoseLandmarker BlazePose model. Models are downloaded (checksum-verified) into
`models/` on first run. Images where the person isn't found are excluded from
the video (use `--keep-missing` to keep them, zoomed on the centre).
Because the face being looked for is known, the search escalates when the strict pass fails: lower detector confidence, larger detection size for small faces, a mirrored image for turned heads, and a relaxed match threshold. `--no-aggressive` disables this.

```
pip install -r requirements.txt
python ken_burns_face_zoom.py photos/ -o out.mp4 --zoom 30 --start-target-size 25 --blur 2 --seconds 3 --order date
```

Options:
- `--zoom PERCENT`: zoom added over each image (default 30)
- `--target chest|face`: zoom on the BlazePose-detected chest (default) or face. The pose crop is guided by the selected face's landmarks and rejects a pose whose nose does not match that face. If BlazePose cannot see both shoulders, chest mode estimates the chest from the identified face.
- `--full-chest`: when both shoulders are confidently visible, zoom from the full image to the complete chest area. Visible hips refine the shoulder-to-hip framing; otherwise it is estimated from shoulder width. This overrides `--zoom` and `--max-upscale`.
- `--start-target-size PERCENT`: initial target height as a percent of frame height (default 25). A distant target starts zoomed in to this size; a target that is already large gets no initial zoom. `0` disables it. `--start-face-size` remains an alias.
- `--max-upscale X`: cap on the initial zoom, as a multiple of the output video resolution (default 1.5), so low-resolution photos are not blown up
- `--scan N`: number of images scanned for people to choose from (default 5)
- `--person K`: number of the person to follow; without it the script writes `people.jpg` (numbered thumbnails) and asks
- `--blur STRENGTH`: blur everything except the selected person's BlazePose segmentation outline; percent of image size (default off). It falls back to the prior face-based approximation only if pose segmentation is unavailable.
- `--seconds N`: seconds per image (default 3)
- `--order random|date|name`: random (`--seed` optional), EXIF capture date, or filename
- `--resolution 480p|720p|1080p|1440p|2160p`: choose a 16:9 output size (default 720p)
- `--width`, `--height`: set a custom output resolution instead of `--resolution`
- `--fps`: output frame rate (default 30)
- `--quality 0-100`: request lower-to-higher encoder quality; support depends on the video writer backend

Images with no detected face zoom on the image centre.

## Setup with make

```
make setup   # creates ./venv and installs requirements.txt
make run     # uses the settings in setup.env (images are read from ./images)
make test
```

`setup.env` is commented and selects `TARGET=chest` or `TARGET=face`. Override
any setting for a single render without editing it, for example:

```
make run ZOOM=50 TARGET=face BLUR=5
make run TARGET=chest FULL_CHEST=1
```

Progress is logged per image (e.g. `image 20 of 245`). Press Ctrl-C while the video is being
rendered to stop early: the video made so far is still saved. The pan eases into position, then
holds while making a small final zoom. Every accepted image logs whether it uses the chest or
face, whether full-chest mode is active, and its initial/final zoom.
