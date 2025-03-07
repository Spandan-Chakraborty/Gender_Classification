import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model and face detector
model = load_model('gender_classifier.keras')

# Check model input shape
print(f"Model expects input shape: {model.input_shape}")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Set expected input size based on model
expected_size = model.input_shape[1:3]  # Extract height & width

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Preprocess face
        face_roi = frame[y:y+h, x:x+w]
        resized = cv2.resize(face_roi, expected_size)  # Resize based on model's input shape
        normalized = resized / 255.0  # Normalize pixel values
        input_tensor = np.expand_dims(normalized, axis=0).astype('float32')  # Ensure correct type

        # Debugging: Print input shape
        print(f"Input tensor shape: {input_tensor.shape}")

        # Prediction
        confidence = model.predict(input_tensor)[0][0]
        gender = 'Female' if confidence >= 0.5 else 'Male'
        color = (0, 255, 0) if gender == 'Male' else (0, 0, 255)

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f'{gender} {confidence:.2f}',  
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.8, color, 2)

    # Display output
    cv2.imshow('Real-time Gender Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
