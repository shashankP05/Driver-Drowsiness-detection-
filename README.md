# Drowsiness & Age Detection System

## Overview
This project provides a real-time, multi-person drowsiness and age detection system using deep learning and computer vision. It features a Tkinter-based GUI for webcam and image upload, and displays per-person drowsiness and age results.

## Features
- Real-time drowsiness detection using a fine-tuned MobileNet model (TensorFlow/Keras)
  - The final saved model used for prediction is `final_eye_state_model_finetuned2.keras` (92% accuracy on validation data)
- Age group detection using a pre-trained Caffe model
- Multi-person support: detects all faces and eyes in the frame
- Timer-based logic: marks as drowsy if both eyes are closed for a set duration
- User-friendly GUI (Tkinter) with webcam and image upload support

## Requirements
- Python 3.7–3.10 (TensorFlow 2.10.1 is not compatible with Python 3.11+)
- See `requirements.txt` for Python dependencies
- Pretrained age detection models:
  - `age_net.caffemodel`
  - `age_deploy.prototxt`

## Installation
1. **Clone or download this repository.**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the age detection models:**
   - Place `age_net.caffemodel` and `age_deploy.prototxt` in the project directory.
   - You can download them from OpenCV's GitHub or other sources:
     - [age_net.caffemodel](https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_net.caffemodel?raw=true)
     - [age_deploy.prototxt](https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_deploy.prototxt?raw=true)

4. **(Optional) Prepare your own dataset:**
   - Place your eye state images in a folder named `Datasets` with subfolders for each class (e.g., `open`, `closed`).

## Usage
### 1. Train the Drowsiness Model
- Open `Drowsiness.ipynb` in VS Code or Jupyter.
- Run all cells to train and evaluate the eye state model.
- The best model will be saved as `final_eye_state_model_finetuned2.keras`.

### 2. Run the GUI
- Make sure `final_eye_state_model_finetuned2.keras`, `age_net.caffemodel`, and `age_deploy.prototxt` are in the project directory.
- Run the GUI:
   ```bash
   python drowsiness_gui.py
   ```
- Use the GUI to start the webcam or upload an image.

## Notes
- The GUI requires a webcam for real-time detection.
- The timer-based logic marks a person as drowsy if both eyes are closed for more than 2 seconds (configurable in the code).
- For best results, use good lighting and face the camera directly.

## Troubleshooting
- If you get errors related to TensorFlow or OpenCV, ensure your Python version is compatible (Python 3.7–3.10).
- If the age detection models are missing, download them as described above.

## Credits
- Eye state model: Trained using MobileNet and Keras.
- Age detection: OpenCV DNN with Caffe models from [learnopencv](https://github.com/spmallick/learnopencv/tree/master/AgeGender).
- The eye state model was trained on the MRL Eye Dataset from Kaggle.

