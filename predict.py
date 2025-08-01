import numpy as np
import cv2
from keras.models import load_model

model = load_model("tf-cnn-model.keras")

def preprocess(img_pil):
    # Convert PIL image to grayscale
    img = img_pil.convert("L")
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to NumPy array
    img_np = np.array(img)
    # Invert colors: white background, black digits
    img_np = 255 - img_np
    # Normalize
    img_np = img_np / 255.0
    # Reshape to (1, 28, 28, 1)
    img_np = img_np.reshape(1, 28, 28, 1)
    return img_np

def predict_digit(img_pil):
    processed = preprocess(img_pil)
    pred = model.predict(processed, verbose=0)[0]
    return np.argmax(pred), float(np.max(pred))

def show_preprocessed(img_pil):
    """Returns the processed image for display (in OpenCV format)."""
    processed = preprocess(img_pil)[0, :, :, 0] * 255
    return processed.astype(np.uint8)
