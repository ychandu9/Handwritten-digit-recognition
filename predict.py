import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model("tf-cnn-model.keras")

def preprocess(img_pil):
    # If image has an alpha (transparency) channel, paste it onto a solid white background
    if img_pil.mode in ("RGBA", "LA") or (img_pil.mode == "P" and "transparency" in img_pil.info):
        background = Image.new("RGB", img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[-1])
        img_pil = background

    # Convert PIL image to grayscale
    img = img_pil.convert("L")
    
    # Convert to NumPy array
    img_np = np.array(img)
    # Invert colors: white background (255) to black (0), black digits (0) to white (255)
    img_np = 255 - img_np

    # Find the bounding box of the drawn digit
    non_zero = np.argwhere(img_np > 50)
    if len(non_zero) == 0:
        # Empty canvas: return blank 28x28 normalized image
        return np.zeros((1, 28, 28, 1), dtype="float32")

    # Get bounding box coordinates
    ymin, xmin = non_zero.min(axis=0)
    ymax, xmax = non_zero.max(axis=0)
    
    # Crop the digit from the inverted image
    digit_crop = img_np[ymin:ymax+1, xmin:xmax+1]
    crop_img = Image.fromarray(digit_crop)
    
    # Scale to fit a 20x20 box (standard MNIST normalization)
    w, h = crop_img.size
    ratio = min(20.0 / w, 20.0 / h)
    new_w = max(1, int(w * ratio))
    new_h = max(1, int(h * ratio))
    
    crop_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_digit = np.array(crop_img)
    
    # Center the 20x20 digit inside a blank 28x28 black canvas
    centered_canvas = np.zeros((28, 28), dtype="uint8")
    offset_y = (28 - new_h) // 2
    offset_x = (28 - new_w) // 2
    centered_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_digit
    
    # Normalize
    img_np = centered_canvas / 255.0
    # Reshape to (1, 28, 28, 1)
    img_np = img_np.reshape(1, 28, 28, 1)
    return img_np

def predict_digit(img_pil):
    processed = preprocess(img_pil)
    pred = model.predict(processed, verbose=0)[0]
    return np.argmax(pred), float(np.max(pred))

def predict_digit_detailed(img_pil):
    """Returns the prediction, confidence, and list of probabilities for all digits (0-9)."""
    processed = preprocess(img_pil)
    pred = model.predict(processed, verbose=0)[0]
    probabilities = [float(p) for p in pred]
    digit = int(np.argmax(pred))
    confidence = float(pred[digit])
    return digit, confidence, probabilities

def show_preprocessed(img_pil):
    """Returns the processed image for display."""
    processed = preprocess(img_pil)[0, :, :, 0] * 255
    return processed.astype(np.uint8)

