import numpy as np
import cv2


def cropImage(image, x, y, width, height, crop_width, crop_height):
	"""
	Dado uma imagem, suas coordenadas iniciais, altura e largura recorta a imagem partindo da coordenada central.
	"""
	x_center = int((2 * x + width)/2)
	y_center = int((2 * y + height)/2)

	crop_width = int(crop_width/2)
	crop_height = int(crop_height/2)

	return image[y_center - crop_height:y_center + crop_height, x_center - crop_width:x_center + crop_width]


def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
	"""
	Dado uma imagem, uma altura e uma largura, re-dimensiona a imagem mantendo a proporção entre altura e largura
	"""
	(h, w) = image.shape[:2]

	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))

	return cv2.resize(image, dim, interpolation=inter)


def rotateImage(image, angle):
	"""
	Dado uma imagem e um ângulo, rotaciona a imagem de acordo com seu centro.
	Método desenvolvido pela Tuany.
	"""

	# Get the image size
	# No that's not an error - NumPy stores image matricies backwards
	image_size = (image.shape[1], image.shape[0])
	image_center = tuple(np.array(image_size) / 2)

	# Convert the OpenCV 3x2 rotation matrix to 3x3
	rot_mat = np.vstack(
		[cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
	)

	rot_mat_notranslate = np.array(rot_mat[0:2, 0:2])

	# Shorthand for below calcs
	image_w2 = image_size[0] * 0.5
	image_h2 = image_size[1] * 0.5

	# Obtain the rotated coordinates of the image corners
	rotated_coords = [
		(np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
		(np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
		(np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
		(np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
	]

	# Find the size of the new image
	x_coords = [pt[0] for pt in rotated_coords]
	x_pos = [x for x in x_coords if x > 0]
	x_neg = [x for x in x_coords if x < 0]

	y_coords = [pt[1] for pt in rotated_coords]
	y_pos = [y for y in y_coords if y > 0]
	y_neg = [y for y in y_coords if y < 0]

	right_bound = max(x_pos)
	left_bound = min(x_neg)
	top_bound = max(y_pos)
	bot_bound = min(y_neg)

	new_w = int(abs(right_bound - left_bound))
	new_h = int(abs(top_bound - bot_bound))

	# We require a translation matrix to keep the image centred
	trans_mat = np.array([
		[1, 0, int(new_w * 0.5 - image_w2)],
		[0, 1, int(new_h * 0.5 - image_h2)],
		[0, 0, 1]
	])

	# Compute the tranform for the combined rotation and translation
	affine_mat = (np.array(trans_mat) * np.array(rot_mat))[0:2, :]

	# Apply the transform
	result = cv2.warpAffine(
		image,
		affine_mat,
		(new_w, new_h),
		flags=cv2.INTER_LINEAR
	)

	return result


def findEyeCoordinates(img, h, w1, w2):
	"""
	Dado uma imagem, uma altura e as laguras, encontra a coordenada dos olhos.
	"""
	sumX = 0
	sumY = 0
	countX = 0
	countY = 0
	for i in range(w1, w2):
		for j in range(0, h):
			px = img[j, i]
			ent = px.astype(np.int)
			if 250 <= ent <= 275:
				sumX = sumX + i
				sumY = sumY + j
				countX = countX + 1
				countY = countY + 1
	x = sumX / countX
	y = sumY / countY
	return x, y
