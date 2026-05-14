# lsm-alphabet-preprocessing
This repository provides the scripts used for video processing, including frame extraction, hand detection with MediaPipe, and automatic generation of 360×360 px crops. The generated crops correspond to the data published on Zenodo and are part of the preprocessing pipeline used in sign language recognition experiments.

---

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe 0.10.14
- NumPy
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

Alternatively, the Conda environment can be recreated using:

```bash
conda env create -f environment.yml
conda activate preprocessing_env
```

---

## MediaPipe Configuration

The following MediaPipe Hands configuration was used:

```python
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

---

## Preprocessing Pipeline

Input videos (~3 seconds, 1920×1080 resolution, 60 fps) are processed as follows:

1. Extract all video frames
2. Detect the hand using MediaPipe Hands (21 landmarks)
3. Compute the hand bounding box from normalized landmark coordinates
4. Estimate the geometric center of the bounding box
5. Extract a centered 360×360 px crop
6. Save crops as JPEG images

Frames without hand detection are discarded (failure rate ≈ 6.6%).

---

## Usage

Example command:

```bash
python processing.py --input ./raw_data --output ./dataset_crops --crop-size 360
```

### Arguments

| Argument | Description |
|---|---|
| `--input` | Directory containing input videos |
| `--output` | Directory where crops will be stored |
| `--crop-size` | Output crop size (default: 360) |
| `--max-hands` | Maximum number of hands to detect (default: 1) |

---

## Output Structure

```text
dataset_crops/
├── video_001/
│   ├── video_001_000000.jpg
│   ├── video_001_000001.jpg
│   └── ...
│
├── video_002/
│   ├── video_002_000000.jpg
│   └── ...
```

---

## Citation

If you use this repository or the generated dataset, please cite:

```bibtex
@article{yourcitation2026,
  title={Your Paper Title},
  author={Author Names},
  journal={Conference or Journal},
  year={2026}
}
```
