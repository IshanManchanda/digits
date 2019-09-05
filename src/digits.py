# from .network import *
import tkinter as tk

from gui import InputGUI


def main():
	# n = NeuralNetwork([784, 256, 10])
	# training, validation, test = load_data()
	# n.train(training[:1000], validation[:500])
	# n.plot()

	root = tk.Tk()
	InputGUI(root)
	root.mainloop()


if __name__ == '__main__':
	main()
