include setup.env

VENV := venv
PY := $(VENV)/bin/python

.PHONY: setup run test clean

setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	python3 -m venv $(VENV)
	$(PY) -m pip install -r requirements.txt pytest
	touch $@

run: setup
	$(PY) ken_burns_face_zoom.py $(INPUT_DIR) -o $(OUTPUT) --order $(ORDER) --zoom $(ZOOM) \
		--target $(TARGET) --start-target-size $(START_TARGET_SIZE) --max-upscale $(MAX_UPSCALE) --scan $(SCAN) $(if $(PERSON),--person $(PERSON)) --blur $(BLUR) --seconds $(SECONDS_PER_IMAGE)

test: setup
	$(PY) -m pytest -q

clean:
	rm -rf $(VENV) .pytest_cache __pycache__
