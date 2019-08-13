from PIL import Image
import numpy as np


def read_files():
	training_files = [f'training_data\\data{i}' for i in range(10)]
	# mydb = connector.connect(
	# 	host="localhost",
	# 	user="root",
	# 	passwd="dps",
	# 	database="digits"
	# )
	# cursor = mydb.cursor()
	# init_db(cursor)
	# mydb.commit()
	cursor = None

	for i in range(10):
		with open(training_files[i], 'rb') as f:
			for j in range(20):
				x = f.read(28 * 28 * 50)
				store_data(i, x, cursor)
				break
		break


def init_db(cursor):
	cursor.execute('SHOW TABLES;')
	tables = cursor.fetchall()

	for i in range(10):
		if (f'training{i}',) in tables:
			cursor.execute(f'DROP TABLE training{i};')

		cursor.execute(f'''
			CREATE TABLE training{i} (
				id INT PRIMARY KEY AUTO_INCREMENT,
				image blob
			);''')


def iter_images(images, size):
	for i in range(0, len(images), size):
		yield images[i:i + size]


def store_data(digit, images, cursor):
	for image in iter_images(images, 28 * 28):
		draw_digit(image)
		cursor.execute(f'INSERT INTO training{digit} (image) VALUES ({str(image)});')
		break


def draw_digit(image):
	arr = np.array(list(image)).reshape((28, 28))
	Image.fromarray(arr).resize((256, 256), Image.ANTIALIAS).show()


read_files()
# init_db()
