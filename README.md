# Conveyor-Belt-Damage-Detection-through-Image-Uploading-Video-Uploading-and-Live-Camera-Detection.


# Conveyor Belt Damage Detection

This repository contains code for detecting conveyor belt damages using a deep learning model. The project supports damage detection through three methods:

1. **Image Upload**: Upload an image to detect damages.
2. **Video Upload**: Upload a video to detect damages frame-by-frame.
3. **Live Camera Feed**: Use a live camera feed for real-time damage detection.

## Features
- **Model Training**: Train a YOLOv8 instance segmentation model on a custom dataset.
- **Image Detection**: Detect damages by uploading an image.
- **Video Detection**: Process video frames to identify damages.
- **Live Detection**: Stream and detect damages from a live camera feed in real-time.

## Folder Structure
```
project-folder/
|-- model_training/       # Code for training the YOLOv8 model
|-- image_detection/      # Code for image-based damage detection
|-- video_detection/      # Code for video-based damage detection
|-- live_camera_feed/     # Code for live camera-based damage detection
|-- models/               # Pretrained or custom-trained YOLOv8 models
|-- requirements.txt      # Python dependencies
|-- README.md             # Project description
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/conveyor-belt-damage-detection.git
   cd conveyor-belt-damage-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the trained model and place it in the `models/` directory.

## Usage

### 1. Image-Based Detection
Run the script to upload an image and detect damages:
```bash
python image_detection/detect_image.py --image-path /path/to/image.jpg
```

### 2. Video-Based Detection
Run the script to upload a video and detect damages frame-by-frame:
```bash
python video_detection/detect_video.py --video-path /path/to/video.mp4
```

### 3. Live Camera Feed Detection
Run the script to stream live camera feed and detect damages:
```bash
python live_camera_feed/detect_live.py
```

### 4. Training the Model
To retrain or fine-tune the YOLOv8 model, use the provided training script:
```bash
python model_training/train_model.py --dataset-path /path/to/dataset
```

## Pretrained Model
- The pretrained model is stored in the `models/` directory.
- To use a custom-trained model, replace the existing model file with your own trained weights.

## Results
The project provides predictions in the form of:
- Annotated images or video frames with bounding boxes.
- Real-time overlays for live camera detection.

## Requirements
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Other dependencies listed in `requirements.txt`

## Acknowledgments
This project uses the YOLOv8 framework for instance segmentation and damage detection.


