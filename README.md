# ken_burns_person_face_zoom

Builds a video from a folder of images. In each image the most prominent face
(largest detected face) is found and a Ken Burns zoom is applied, centred on it.

```
pip install -r requirements.txt
python ken_burns_face_zoom.py photos/ -o out.mp4 --zoom 30 --start-zoom 10 --blur 2 --seconds 3 --order date
```

Options:
- `--zoom PERCENT`: zoom added over each image (default 30)
- `--start-zoom PERCENT`: initial zoom at the start of each image (default 0)
- `--blur STRENGTH`: blur everything except the subject; percent of image size (default off)
- `--seconds N`: seconds per image (default 3)
- `--order random|date|name`: random (`--seed` optional), EXIF capture date, or filename
- `--fps`, `--width`, `--height`: output video settings

Images with no detected face zoom on the image centre.
