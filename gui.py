import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
from predict import predict_digit, show_preprocessed

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas_width = 200
        self.canvas_height = 200

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 20))
        self.label.pack()

        self.pre_img_label = tk.Label(self)
        self.pre_img_label.pack()

        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.image1 = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image1)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<ButtonRelease-1>", lambda event: self.predict())

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")
        self.label.config(text="Draw a digit")
        self.pre_img_label.config(image="")

    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def predict(self):
        digit, confidence = predict_digit(self.image1)
        self.label.config(text=f"Prediction: {digit}, Confidence: {confidence:.2f}")

        pre_img = show_preprocessed(self.image1)
        # Convert to PIL image to display
        img = Image.fromarray(pre_img)
        img = img.resize((100, 100), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(img)
        self.pre_img_label.configure(image=tk_img)
        self.pre_img_label.image = tk_img

if __name__ == "__main__":
    app = App()
    app.mainloop()
