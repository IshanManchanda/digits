import tkinter as tk

import numpy as np

from network import draw_digit
from preprocessor import dots_to_image, dots_to_image2, render_digit, \
	deskew_image

desc = 'Draw a single digit in the canvas.\n' + \
       'For best output, try to ensure it is centered\n' + \
       'in the frame and nearly fills it.'


class InputGUI:
	def __init__(self, root, n=None):
		self.root = root
		self.n = n
		self.dots = []
		self.scale = 10
		self.size = 27 * self.scale

		self.root.minsize(700, 400)
		self.root.title("pyDigits")
		self.label = tk.Label(root, text=desc)
		self.label.grid(column=1, row=0, columnspan=3)

		self.canvas = tk.Canvas(
			self.root, width=self.size, height=self.size,
			highlightthickness=2, highlightbackground='black'
		)
		self.canvas.grid(column=1, row=1, columnspan=3)
		self.canvas.bind('<Button-1>', self.draw)
		self.canvas.bind('<B1-Motion>', self.draw)

		self.clear_button = tk.Button(
			self.root, text="Clear", command=self.clear
		)
		self.predict_button = tk.Button(
			self.root, text='Predict', command=self.predict
		)
		self.close_button = tk.Button(
			self.root, text="Close", command=self.root.destroy
		)

		self.clear_button.grid(column=1, row=2)
		self.predict_button.grid(column=2, row=2)
		self.close_button.grid(column=3, row=2)

		self.root.grid_columnconfigure(0, weight=1)
		self.root.grid_columnconfigure(4, weight=1)

	def clear(self):
		self.canvas.delete("all")
		self.dots = []

	def predict(self):
		print('Number of dots:', len(self.dots))
		data = dots_to_image2(self.dots, self.scale)
		render_digit(data * 255)

		deskewed = deskew_image(data)
		render_digit(deskewed)

		if self.n:
			prediction = self.n.predict(data)
			digit = np.argmax(prediction)
			print(prediction, digit, prediction[digit])

	def draw(self, event):
		x, y = event.x, event.y
		if 0 <= x < 28 * self.scale and 0 <= y < 28 * self.scale:
			self.dots.append((x, y))
			self.canvas.create_oval(
				x - self.scale * 1, y - self.scale * 1,
				x + self.scale * 1, y + self.scale * 1,
				fill='#333333'
			)


def gui():
	root = tk.Tk()
	InputGUI(root)
	root.mainloop()


def process_dots(dots, scale):
	data = np.zeros((28, 28))

	for dot in dots:
		x, y = dot[0] // scale, dot[1] // scale
		data[y][x] = 1

		# for x1, y1 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
		# 	data[x1, y1] += 2
		#
		# data[x - 1, y - 1] += 1
		# data[x - 1, y + 1] += 1
		# data[x + 1, y - 1] += 1
		# data[x + 1, y + 1] += 1

	# data /= data.max()
	# print(data)
	return data.flatten()


if __name__ == '__main__':
	gui()
