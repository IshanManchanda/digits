import tkinter as tk

import numpy as np

from network import NeuralNetwork, draw_digit

desc = 'Draw a single digit in the canvas.\n' + \
       'For best output, try to ensure it is centered\n' + \
       'in the frame and nearly fills it.'


class InputGUI:
	def __init__(self, root, n=None):
		self.root = root
		self.n = n
		self.dots = []
		self.scale = 10
		self.size = 28 * self.scale

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
		print(len(self.dots))
		print(self.dots)
		data = process_dots(self.dots, self.scale)
		draw_digit(data)

		if self.n:
			self.n.predict(process_dots(self.dots, self.scale))

	def draw(self, event):
		x, y = event.x, event.y
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
		data[y][x] += 4

		# for x1, y1 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
		# 	data[x1, y1] += 2

		# for x1, y1 in (
		# 	(x - 1, y - 1), (x - 1, y + 1),
		# 	(x + 1, y - 1), (x + 1, y + 1)):
		# 	data[x1, y1] += 1
		# data[x - 1, y - 1] += 1
		# data[x - 1, y + 1] += 1
		# data[x + 1, y - 1] += 1
		# data[x + 1, y + 1] += 1

	data *= 255 / 16
	return data


if __name__ == '__main__':
	gui()
