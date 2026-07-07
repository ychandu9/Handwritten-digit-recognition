# Handwritten Digit Recognition with GUI (MNIST)

This is a Python project that allows you to draw digits on a GUI canvas and predicts the digit using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 📁 Files

- `train_model.py` – Trains a CNN model on the MNIST dataset and saves it as `tf-cnn-model.keras`
- `predict.py` – Loads the trained model, preprocesses input, and predicts the digit
- `gui.py` – Provides a Tkinter-based drawing canvas for digit input and live predictions

---

## 🚀 Requirements

Install dependencies in a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
