import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import interpolation


def dots_to_image(dots, scale):
	image = Image.new('L', (280, 280))
	draw = ImageDraw.Draw(image)
	for x, y in dots:
		draw.ellipse((x - scale, y - scale, x + scale, y + scale), fill=255)

	# Since size_normalize accepts and return a np array, we convert
	# the argument and the return value accordingly
	image = Image.fromarray(size_normalize(np.array(image)))
	return np.array(image.resize((28, 28), Image.ANTIALIAS)) / 255


def draw_digit(image):
	arr = (np.array(image).reshape((28, 28)) * 255).astype('uint8')
	Image.fromarray(arr).resize((256, 256), Image.ANTIALIAS).show()


def size_normalize(image):
	# First, we strip the empty space around the image (top and bottom)
	i, j = 0, -1
	while np.sum(image[i]) == 0:
		i += 1
	while np.sum(image[j]) == 0:
		j -= 1
	image = image[i:j]

	# Similarly for the sides
	while np.sum(image[:, 0]) == 0:
		image = np.delete(image, 0, 1)

	while np.sum(image[:, -1]) == 0:
		image = np.delete(image, -1, 1)

	# Now, we want the image to fit a 20x20 box. Thus, we fit the max of the 2
	# dimensions to this desired size and scale the other accordingly.
	# Note that numpy uses row-major ordering...
	rows, cols = image.shape
	scale = 200 / max(rows, cols)
	rows, cols = round(rows * scale), round(cols * scale)

	# ...but PIL uses column-major.
	image = np.array(Image.fromarray(image).resize(
		(cols, rows), Image.ANTIALIAS
	))

	# Finally, we need to pad our image to bring the size up to 28x28.
	padding_row = (
		int(np.ceil((280 - rows) / 2)),
		int(np.floor((280 - rows) / 2))
	)
	padding_col = (
		int(np.ceil((280 - cols) / 2)),
		int(np.floor((280 - cols) / 2))
	)
	padding = (padding_row, padding_col)
	return np.pad(image, padding, 'constant', constant_values=(0,))


def compute_moments(image):
	# This function computes the center of mass and the covariance of the image
	sum_of_pixels = np.sum(image)

	# We create a mesh grid which will allow us to extract the x and y
	# coordinates separately
	c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]

	# Center of mass of the image about x and y axes
	mu_x = np.sum(c0 * image) / sum_of_pixels
	mu_y = np.sum(c1 * image) / sum_of_pixels
	offset_x = c0 - mu_x
	offset_y = c1 - mu_y

	# Variance about the axes
	variance_x = np.sum(offset_x ** 2 * image) / sum_of_pixels
	variance_y = np.sum(offset_y ** 2 * image) / sum_of_pixels

	# The covariance is a measure of how "related" 2 quantities are.
	# If one increases with the other, they show a positive covariance.
	covariance = np.sum(offset_x * offset_y * image) / sum_of_pixels

	# The covariance matrix for 2 quantities is a n * n matrix.
	# The (i, j)th cell holds the covariance between the i'th and the j'th
	# quantities. In particular, when i = j, the cell simply holds the variance
	# of the i'th quantity. Further, (i, j)th element = (j, i)th element.
	covariance_matrix = np.array([
		[variance_x, covariance],
		[covariance, variance_y]
	])
	mu_vector = np.array([mu_x, mu_y])
	return mu_vector, covariance_matrix


def deskew_image(image):
	center_of_mass, covariance_matrix = compute_moments(image)
	# Alpha is the ratio of the covariance and the variance
	alpha = covariance_matrix[0, 1] / covariance_matrix[0, 0]

	# The affine matrix, which numerically represents the transformation
	affine = np.array([[1, 0], [alpha, 1]])

	# We translate the image so as to bring the center of mass to the center
	offset = center_of_mass - np.dot(affine, np.array(image.shape) / 2)
	deskewed = interpolation.affine_transform(image, affine, offset=offset)

	# The image needs to be renormalized after the transformation
	return (deskewed - deskewed.min()) / (deskewed.max() - deskewed.min())
