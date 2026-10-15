
"""
MNIST Digit Recognizer - desktop app.
 
Features
  * Loads the trained model (mnist_cnn.keras); trains one automatically if missing
  * Single-digit mode: draw / open one digit, see top-3 predictions
  * Multi-digit mode: draw / open several digits, get the full number
  * Open Image button to recognize digits from a picture file
 
Install:  pip install -r requirements.txt
Run:      python app.py
"""
 
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog
 
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageTk
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
 
MODEL_PATH = "mnist_cnn.keras"
W, H = 700, 260
 
 
# ===============================================================
# MODEL: build / train
# (same architecture as train.py)
# ===============================================================
def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
 
 
def train_model(on_epoch=None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
 
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    if on_epoch:
        cbs.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, logs: on_epoch(
                f"Training... epoch {e + 1}/15  val_acc={logs['val_accuracy']:.4f}")))
 
    model = build_model()
    model.fit(x_train, y_train, validation_split=0.1, epochs=15,
              batch_size=128, callbacks=cbs, verbose=2)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    return model, acc
 
 
# ===============================================================
# IMAGE PROCESSING: preprocess + segmentation
# ===============================================================
def to_white_on_black(img):
    g = img.convert("L")
    if np.array(g).mean() > 127:
        g = ImageOps.invert(g)
    return g
 
 
def preprocess(img):
    """Single digit image -> (1, 28, 28, 1) MNIST-style array."""
    img = to_white_on_black(img)
    bbox = img.point(lambda p: 255 if p > 30 else 0).getbbox()
    if bbox is None:
        raise ValueError("No digit found.")
    img = img.crop(bbox)
    w, h = img.size
    s = 20.0 / max(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2))
    return (np.array(canvas).astype("float32") / 255.0)[np.newaxis, ..., np.newaxis]
 
 
def segment_digits(img):
    """Return boxes (x1, y1, x2, y2) of each digit, left to right."""
    arr = np.array(to_white_on_black(img))
    mask = arr > max(30, 0.35 * arr.max())
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    boxes = [[s[1].start, s[0].start, s[1].stop, s[0].stop]
             for s in ndimage.find_objects(labeled)]
    max_h = max(b[3] - b[1] for b in boxes)
    boxes = [b for b in boxes if (b[3] - b[1]) >= 0.25 * max_h]   # drop specks
    boxes.sort(key=lambda b: b[0])
    merged = [boxes[0]]
    for b in boxes[1:]:                                           # merge overlaps
        last = merged[-1]
        if b[0] <= last[2]:
            last[0], last[1] = min(last[0], b[0]), min(last[1], b[1])
            last[2], last[3] = max(last[2], b[2]), max(last[3], b[3])
        else:
            merged.append(b)
    return [tuple(b) for b in merged]
 
 
# ===============================================================
# APP
# ===============================================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("MNIST Digit Recognizer")
        self.model = None
        self.q = queue.Queue()
        self.mode = tk.StringVar(value="multi")
        self.loaded_image = False      # True when a file is shown on the canvas
        self.photo = None
 
        self.canvas = tk.Canvas(root, width=W, height=H, bg="black",
                                highlightthickness=1)
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
 
        self.img = Image.new("L", (W, H), 0)
        self.draw = ImageDraw.Draw(self.img)
 
        modes = tk.Frame(root)
        modes.pack()
        tk.Radiobutton(modes, text="Single digit", variable=self.mode,
                       value="single").pack(side="left", padx=8)
        tk.Radiobutton(modes, text="Multiple digits", variable=self.mode,
                       value="multi").pack(side="left", padx=8)
 
        self.result = tk.Label(root, text="", font=("Arial", 16, "bold"))
        self.result.pack(pady=(8, 0))
        self.status = tk.Label(root, text="", fg="gray")
        self.status.pack()
 
        btns = tk.Frame(root)
        btns.pack(pady=10)
        self.buttons = []
        for text, cmd in [("Predict", self.predict), ("Clear", self.clear),
                          ("Open Image...", self.open_image),
                          ("Retrain Model", self.start_training)]:
            b = tk.Button(btns, text=text, width=13, command=cmd)
            b.pack(side="left", padx=4)
            self.buttons.append(b)
 
        if os.path.exists(MODEL_PATH):
            self.model = keras.models.load_model(MODEL_PATH)
            self.status.config(text="Model loaded. Draw digits and click Predict.")
        else:
            self.start_training()
        root.after(200, self.poll)
 
    # ---------- training (background thread) ----------
    def start_training(self):
        self.set_buttons(False)
        self.status.config(text="Downloading data / training... (1-3 min)")
 
        def work():
            try:
                model, acc = train_model(lambda m: self.q.put(("status", m)))
                self.q.put(("done", (model, acc)))
            except Exception as e:
                self.q.put(("error", str(e)))
 
        threading.Thread(target=work, daemon=True).start()
 
    def poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "status":
                    self.status.config(text=data)
                elif kind == "done":
                    self.model, acc = data
                    self.status.config(text=f"Training done. Test accuracy: {acc:.2%}")
                    self.set_buttons(True)
                elif kind == "error":
                    self.status.config(text=f"Error: {data}")
                    self.set_buttons(True)
        except queue.Empty:
            pass
        self.root.after(200, self.poll)
 
    def set_buttons(self, enabled):
        for b in self.buttons:
            b.config(state="normal" if enabled else "disabled")
 
    # ---------- drawing ----------
    def paint(self, e):
        if self.loaded_image:
            return
        r = 8
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r,
                                fill="white", outline="white", tags="ink")
        self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)
 
    def clear(self):
        self.canvas.delete("all")
        self.img = Image.new("L", (W, H), 0)
        self.draw = ImageDraw.Draw(self.img)
        self.loaded_image = False
        self.result.config(text="")
 
    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not path:
            return
        self.clear()
        img = Image.open(path).convert("L")
        img.thumbnail((W, H))
        self.img = img
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.loaded_image = True
        self.predict()
 
    # ---------- prediction ----------
    def predict(self):
        if self.model is None:
            self.result.config(text="Model not ready yet")
            return
        self.canvas.delete("box")
        try:
            if self.mode.get() == "single":
                probs = self.model.predict(preprocess(self.img), verbose=0)[0]
                top = probs.argsort()[::-1][:3]
                self.result.config(text=f"Digit: {top[0]}   ({probs[top[0]]:.1%})")
                self.status.config(text="Top 3: " + ", ".join(
                    f"{d} ({probs[d]:.1%})" for d in top))
            else:
                boxes = segment_digits(self.img)
                if not boxes:
                    raise ValueError("No digits found.")
                batch = []
                for (x1, y1, x2, y2) in boxes:
                    crop = self.img.crop((max(0, x1 - 4), max(0, y1 - 4),
                                          x2 + 4, y2 + 4))
                    batch.append(preprocess(crop)[0])
                probs = self.model.predict(np.stack(batch), verbose=0)
                digits = probs.argmax(axis=1)
                confs = probs.max(axis=1)
                for (x1, y1, x2, y2), d, c in zip(boxes, digits, confs):
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="red",
                                                 width=2, tags="box")
                    self.canvas.create_text(x1 + 2, max(8, y1 - 8), text=f"{d}",
                                            fill="yellow", anchor="w",
                                            font=("Arial", 12, "bold"), tags="box")
                self.result.config(text="Number: " + "".join(map(str, digits)))
                self.status.config(text="Confidence per digit: " + ", ".join(
                    f"{d}:{c:.0%}" for d, c in zip(digits, confs)))
        except ValueError as err:
            self.result.config(text=str(err))
 
 
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
 






