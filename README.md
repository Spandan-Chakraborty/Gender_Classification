# Real-time Gender Detection using OpenCV and TensorFlow

## Overview
This project performs real-time gender classification using a pre-trained deep learning model and OpenCV for face detection. The model predicts gender based on facial features captured through a webcam.

## Features
- Uses a pre-trained TensorFlow/Keras model for gender classification.
- Real-time face detection using OpenCV's Haar cascades.
- Normalizes and resizes detected faces to match the model's input shape.
- Displays real-time predictions with bounding boxes and confidence scores.
- Simple and easy-to-use Python implementation.

## Requirements
Make sure you have the following dependencies installed:

```bash
pip install opencv-python numpy tensorflow
```

## File Structure
```
|-- gender_classifier.keras  # Pre-trained gender classification model
|-- gender_detection.py      # Main script for real-time gender detection
|-- README.md                # Documentation file
```

## Usage
### 1. Load the Model and Start Webcam
Run the following command to start real-time gender detection:

```bash
python gender_detection.py
```

### 2. Key Controls
- Press `q` to exit the program.

## How It Works
1. **Face Detection:**
   - The script captures frames from the webcam.
   - Converts the frame to grayscale and detects faces using OpenCV's Haar cascade classifier.

2. **Preprocessing:**
   - Extracts and resizes detected face regions to match the model's expected input shape.
   - Normalizes pixel values.

3. **Prediction:**
   - Feeds the processed face into the deep learning model.
   - Predicts gender (`Male` or `Female`) with confidence scores.

4. **Displaying Results:**
   - Draws bounding boxes around detected faces.
   - Displays gender classification results and confidence scores in real-time.

## Model Details
- The model is trained using TensorFlow/Keras and expects input images in a specific size.
- Ensure `gender_classifier.keras` is placed in the same directory as the script.

## Troubleshooting
- If the model file is missing, ensure `gender_classifier.keras` is present.
- If the webcam doesn't initialize, check if another application is using it.
- If OpenCV's Haar cascade fails, try reinstalling OpenCV:
  ```bash
  pip install --upgrade opencv-python
  ```

## Future Improvements
- Improve accuracy with a more robust model (e.g., CNN-based classifiers).
- Optimize face detection using deep learning-based methods like MTCNN or Dlib.
- Expand the dataset for better generalization.

## License
This project is open-source under the MIT License.

