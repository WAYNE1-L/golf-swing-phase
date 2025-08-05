# 🏌️ Golf Swing Phase Analysis

A deep learning-based system for **golf swing phase recognition**, using pose estimation and LSTM models.

## 📦 Project Structure

```
.
├── data/                   # All input video, image and CSV data
│   ├── raw_videos/        # Raw swing videos (MP4)
│   ├── keypoints_csv/     # Exported pose keypoints (from MediaPipe)
│   ├── labeled_csv/       # Labeled swing data (with phases)
│   ├── images/            # Preview and input frames
│   └── npz/               # Prepared training data in numpy format
│
├── model/                 # Trained models and output scores
├── results/               # Inference results and evaluation charts
├── scripts/               # All logic scripts grouped by task
│   ├── inference/         # Phase detection, similarity scoring
│   ├── preprocessing/     # Data preparation, CSV batch export
│   ├── training/          # LSTM training code
│   └── gui/               # Labeling GUI (PyQt)
│
├── phase_comments.txt     # Example output (phase recognition + comments)
├── upload_to_github.py    # Upload automation
└── README.md              # Project documentation
```

## 🔍 Features

- 🎯 Recognizes 5 swing phases: `Setup`, `Takeaway`, `Backswing`, `Downswing`, `Follow-through`
- 📈 Trained using LSTM with 66D input (33 keypoints × x/y)
- 🖼️ Pose extraction via MediaPipe + OpenCV
- ✅ Output commentary per frame using templates
- 📊 Includes similarity scoring and elbow angle visualization

## 🛠️ How to Use

### 1. Prepare Input

Put your `.mp4` swing video into:
```bash
data/raw_videos/
```

Then extract pose keypoints:
```bash
python scripts/preprocessing/golf_pose_export_full.py --input swing1.mp4 --output swing1.csv
```

### 2. Label & Train

```bash
python scripts/gui/label_swing_gui.py  # Label phase per frame
python scripts/preprocessing/make_train_dataset.py  # Convert to npz
python scripts/training/train_phase_lstm.py         # Train LSTM model
```

### 3. Predict & Comment

```bash
python scripts/inference/phase_inference.py  # Predict and generate comments
```

## 📌 Requirements

```bash
pip install -r requirements.txt
```

Recommended packages:
- `tensorflow`
- `numpy`
- `opencv-python`
- `pandas`
- `mediapipe`

## 📜 License

MIT License (or your choice)
