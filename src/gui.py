import tkinter as tk

desc = 'Draw a single digit in the canvas.\n' + \
       'For best output, try to ensure it is centered\n' + \
       'in the frame and nearly fills it.'


class InputGUI:
	def __init__(self, root: tk.Tk):
		self.root = root
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
		print(self.canvas.find_all())
		print(len(self.dots))
		print(self.dots)

	def draw(self, event):
		x, y = event.x, event.y
		self.dots.append((x, y))
		self.canvas.create_oval(
			x - self.scale / 2, y - self.scale / 2,
			x + self.scale / 2, y + self.scale / 2,
			fill='black'
		)


def main():
	root = tk.Tk()
	gui = InputGUI(root)
	root.mainloop()


if __name__ == '__main__':
	main()
