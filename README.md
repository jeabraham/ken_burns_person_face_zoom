# ken_burns_person_face_zoom

Builds a video from a folder of images. The first N images are scanned for
people; you pick which person to follow (from numbered thumbnails in
`people.jpg`, or with `--person K`). In every image that person's face is
found and a Ken Burns zoom is applied, centred on it. Face detection and
recognition use OpenCV's YuNet and SFace models, downloaded (checksum-verified)
into `models/` on first run. Images where the person isn't found zoom on the centre.

```
pip install -r requirements.txt
python ken_burns_face_zoom.py photos/ -o out.mp4 --zoom 30 --start-face-size 25 --blur 2 --seconds 3 --order date
```

Options:
- `--zoom PERCENT`: zoom added over each image (default 30)
- `--start-face-size PERCENT`: initial face height as a percent of frame height (default 25). A distant face in a group shot starts zoomed in to this size; a face that is already large (e.g. a portrait) gets no initial zoom. `0` disables it
- `--max-upscale X`: cap on the initial zoom, as a multiple of the output video resolution (default 1.5), so low-resolution photos are not blown up
- `--scan N`: number of images scanned for people to choose from (default 5)
- `--person K`: number of the person to follow; without it the script writes `people.jpg` (numbered thumbnails) and asks
- `--blur STRENGTH`: blur everything except the subject; percent of image size (default off)
- `--seconds N`: seconds per image (default 3)
- `--order random|date|name`: random (`--seed` optional), EXIF capture date, or filename
- `--fps`, `--width`, `--height`: output video settings

Images with no detected face zoom on the image centre.

## Setup with make

```
make setup   # creates ./venv and installs requirements.txt
make run     # uses the settings in setup.env (images are read from ./images)
make test
```
