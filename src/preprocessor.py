import numpy as np
from PIL import Image
from scipy.ndimage import interpolation


def dots_to_image(dots, scale):
	data = np.zeros((28, 28))

	for dot in dots:
		x, y = dot[0] // scale, dot[1] // scale
		data[y][x] = 10

		for x1, y1 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
			data[y1, x1] += 2

		# data[y - 1, x - 1] += 1
		# data[y - 1, x + 1] += 1
		# data[y + 1, x - 1] += 1
		# data[y + 1, x + 1] += 1

	data = np.minimum(data * 2 / data.max(), 1)
	# return data.flatten()
	return data


def dots_to_image2(dots, scale):
	data = np.zeros((28, 28))

	for dot in dots:
		x, y = round(dot[0] / scale), round(dot[1] / scale)
		data[y][x] = 4

		# for x1, y1 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
		# 	data[y1, x1] += 2

		# data[y - 1, x - 1] += 1
		# data[y - 1, x + 1] += 1
		# data[y + 1, x - 1] += 1
		# data[y + 1, x + 1] += 1

	data = np.minimum(data * 2 / data.max(), 1)
	# return data.flatten()
	return data


def render_digit(image):
	arr = np.array(image).reshape((28, 28)) * 256
	Image.fromarray(arr).resize((256, 256), Image.ANTIALIAS).show()


def size_normalize(image: np.ndarray):
	# To size normalize, we calculate the bounding box of the drawn digit
	points = image.nonzero()
	height = max(points[0]) - min(points[0])
	width = max(points[1]) - min(points[1])


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
