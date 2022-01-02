import os
import tkinter as tk

import numpy as np

from src.globals import current_dir
from src.preprocessor import deskew_image, dots_to_image
from src.utils import draw_digit, save_digit


class InputGUI:
    def __init__(self, root, n=None):
        self.root = root
        self.n = n  # NeuralNetwork object
        self.dots = []  # Array to store locations of dots
        self.scale = 15  # Stroke size
        self.size = 28 * 10 - 1

        self.root.minsize(700, 400)
        self.root.title('Digits')

        desc = \
            'Draw a single digit in the canvas.\n' + \
            'For best output, try to ensure it is centered\n' + \
            'in the frame and nearly fills it.'
        self.label_desc = tk.Label(root, text=desc)
        self.label_desc.grid(column=1, row=0, columnspan=3)

        self.label_prediction = tk.Label(root, text='')
        self.label_prediction.grid(column=1, row=2)

        self.label_confidence = tk.Label(root, text='')
        self.label_confidence.grid(column=2, row=2)

        # TODO: Add text field(s) to display prediction and confidence
        # REVIEW: Perhaps allow user to enter correct answer and save
        #  as additional test data?
        #  Perhaps make new training data out of it?
        self.canvas = tk.Canvas(
            self.root, width=self.size, height=self.size,
            highlightthickness=2, highlightbackground='black'
        )
        self.canvas.grid(column=1, row=1, columnspan=3)
        self.canvas.bind('<Button-1>', self.draw)
        self.canvas.bind('<B1-Motion>', self.draw)

        self.clear_button = tk.Button(
            self.root, text='Clear', command=self.clear
        )
        self.predict_button = tk.Button(
            self.root, text='Predict', command=self.predict
        )
        self.close_button = tk.Button(
            self.root, text='Close', command=self.root.destroy
        )

        self.clear_button.grid(column=1, row=3)
        self.predict_button.grid(column=2, row=3)
        self.close_button.grid(column=3, row=3)

        # Needed for center alignment
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(4, weight=1)

    def clear(self):
        self.canvas.delete('all')
        self.dots = []

    def predict(self):
        prediction_dir = get_prediction_dir()

        data = dots_to_image(self.dots, self.scale)
        deskewed = deskew_image(data)

        save_digit(data, os.path.join(prediction_dir, 'raw.png'))
        save_digit(deskewed, os.path.join(prediction_dir, 'deskewed.png'))

        # DEBUG: Draw digits to check deskewing
        draw_digit(data)
        draw_digit(deskewed)

        if self.n:
            # TODO: Save raw as well as deskewed prediction as json
            # prediction = self.n.predict(data.reshape((784,)))
            prediction = self.n.predict(deskewed.reshape((784,)))
            digit = np.argmax(prediction)
            print('Prediction: %s, confidence: %d%%' % (
                digit, prediction[digit] * 100
            ))
            print(prediction)

    def draw(self, event):
        x, y = event.x, event.y
        if 0 <= x < self.size and 0 <= y < self.size:
            self.dots.append((x, y))
            self.canvas.create_oval(
                x - self.scale, y - self.scale,
                x + self.scale, y + self.scale,
                fill='#222222'
            )


def run_gui(n=None):
    root = tk.Tk()
    InputGUI(root, n)
    root.mainloop()


def get_prediction_dir():
    prediction_dir = os.path.join(current_dir, 'predictions')
    if not os.path.isdir(prediction_dir):
        os.mkdir(prediction_dir)

    i = 1
    while os.path.isdir(os.path.join(prediction_dir, str(i))):
        i += 1

    path = os.path.join(prediction_dir, str(i))
    os.mkdir(path)
    return path


if __name__ == '__main__':
    run_gui()
