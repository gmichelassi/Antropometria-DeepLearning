import cv2
import numpy as np
from PIL import Image


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
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


def rotate_image(image, angle):
	"""
	Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
	(in degrees). The returned image will be large enough to hold the entire
	new image, with a black background
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
		(np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
		(np.array([image_w2,  image_h2]) * rot_mat_notranslate).A[0],
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


def eye_coordinate(img, h, w1, w2):
	sumX, sumY = 0, 0
	countX, countY = 0, 0
	for i in range(math.floor(w1), math.floor(w2)):
		for j in range(0, h):
			px = img[j, i]
			ent = px.astype(np.int)
			if (ent <= 275) & (ent >= 250):
				sumX = sumX + i
				sumY = sumY + j
				countX = countX + 1
				countY = countY + 1
	x = sumX / countX
	y = sumY / countY
	return x, y


def preprocess_image():
	img = cv2.imread('img/MIT-10.jpg')

	face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
	eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

	# parâmetros abaixo são: scale factor, min_neighbors, scale_image e min_size
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20, 20))

	for (x, y, width, height) in faces:
		cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)

		# Recortar somente a face
		image = Image.open('img/MIT-10.jpg')
		detected_face = image.crop((x, y, x + width, y + height))
		face = np.asarray(detected_face)
		face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

		# roi stands for "region of interest"
		# roi_gray = gray_img[y:y + height, x: x + width]
		roi_colorful = img[y:y + height, x: x + width]

		# Encontrar os olhos
		eyes = eye_cascade.detectMultiScale(face_gray)
		for (eye_x, eye_y, eye_width, eye_height) in eyes:
			cv2.rectangle(roi_colorful, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)

			eyes_pair = face_gray[eye_y:eye_y + eye_height, eye_x:eye_x + eye_width]
			canny = cv2.Canny(eyes_pair, 50, 245)
			kernel = np.ones((3, 3), np.uint8)
			gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)

			# find coordinates of the pupils
			h, w = gradient.shape[0], gradient.shape[1]

			x1, y1 = eye_coordinate(gradient, h, 0, w / 2)  # left eye
			x1, y1 = abs(x1), abs(y1)  # normalizar geometricamente

			x2, y2 = eye_coordinate(gradient, h, w / 2, w)  # right eye
			x2, y2 = abs(x2), abs(y2)

			dx, dy = abs(x2 - x1), abs(y2 - y1) * -1

			z = Decimal(dy) / Decimal(dx)
			alpha_complex = cmath.atan(z)
			alpha = cmath.phase(alpha_complex)
			alpha = alpha / 2

	resized_img = ResizeWithAspectRatio(img, width=500)
	cv2.imshow("Image", resized_img)
	cv2.waitKey(0)


if __name__ == '__main__':
	preprocess_image()
