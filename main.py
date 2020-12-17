import cv2
import numpy as np
from PIL import Image


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
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
		roi_gray = gray_img[y:y + height, x: x + width]
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

			y1, x1 = eye_coordinate(gradient, h, 0, w / 2)  # left eye
			x1, y1 = abs(x1), abs(y1)  # normalizar geometricamente

			y2, x2 = eye_coordinate(gradient, h, w / 2, w)  # right eye
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
