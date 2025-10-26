import tkinter as tk

from PIL import Image, ImageDraw

# === CORE !!! ===
import core

# === Глобальні змінні ===
mode = "train"
TRAIN_COUNT = 6

# === INTERFACE ===
# === ROOT ===
root = tk.Tk()
root.title("🧠 Digit Recognizer Trainer")
root.resizable(False, False)
root.columnconfigure(0, weight=1)
root.protocol("WM_DELETE_WINDOW", lambda: 0)
# === CONTROLS ===
canvas = tk.Canvas(root)
status_label = tk.Label(root, text="Готово ✅")
entry = tk.Entry(root)
# === Drawing canvas ===
image = Image.new("L", (280, 280), "white")
draw = ImageDraw.Draw(image)
# === BUTTONS ===
mode_btn = tk.Button(root, text="TRAIN")
ok_btn = tk.Button(root, text="OK")
clear_btn = tk.Button(root, text="Очистити")
quit_btn = tk.Button(root, text="Вихід")


# === LISTENERS ===
def paint(event):
    x, y = event.x, event.y
    r = 8
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black')
    draw.ellipse((x - r, y - r, x + r, y + r), fill='black')


# === Очищення ===
def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")


# === Перемикання режиму ===
def toggle_mode():
    global mode
    mode = "predict" if mode == "train" else "train"
    mode_btn.config(text=f"{mode.upper()}")
    clear_canvas()


# === Обробка зображення ===
def process_image_listener():
    img = image.resize((28, 28))
    if mode == "train":
        if entry.get().isdigit():
            label_value = int(entry.get())
            status_label.config(text="🔄 Навчаюся...")
            root.update()
            # TRAIN
            trained = core.train(img, label_value, TRAIN_COUNT)
            if trained:
                status_label.config(text="✅ Модель оновлена і збережена")
                root.update()
        entry.delete(0, tk.END)
    else:  # predict
        result = core.predict(img)
        entry.delete(0, tk.END)
        entry.insert(0, f"{result}")
    clear_canvas()


# === LISTENERS ===
canvas.bind("<B1-Motion>", paint)
mode_btn.config(command=toggle_mode)
ok_btn.config(command=process_image_listener)
clear_btn.config(command=clear_canvas)
quit_btn.config(command=root.destroy)
# === STYLE ===
WIDTH = 12
FONT = ("serif", 12, "normal")
FONT_LARGE = ("serif", 25)
#
canvas.config(width=280, height=280, bg="white")
status_label.config(fg="green")
entry.config(width=WIDTH, font=FONT_LARGE, justify=tk.CENTER)
mode_btn.config(bg="lightblue", font=FONT, width=WIDTH)
ok_btn.config(bg="lightgreen", font=FONT, width=WIDTH)
clear_btn.config(bg="#DCDCDC", font=FONT, width=WIDTH)
quit_btn.config(bg="salmon", font=FONT, width=WIDTH)
# === POSITION ===
quit_btn.grid(row=0, column=0)
mode_btn.grid(row=0, column=1)
status_label.grid(row=1, column=0, columnspan=2, pady=10)
canvas.grid(row=2, column=0, columnspan=2)
entry.grid(row=3, column=0, pady=15, columnspan=2)
clear_btn.grid(row=4, column=0)
ok_btn.grid(row=4, column=1)
# === WINDOW POSITION ===
root.update_idletasks()
window_width = root.winfo_width()
window_height = root.winfo_height()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x}+{y - 100}")
# === START ===
root.mainloop()
