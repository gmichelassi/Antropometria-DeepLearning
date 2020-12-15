import cv2


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


def load_image():
	img = cv2.imread('img/MIT-10.jpg')

	face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
	eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

	# parâmetros abaixo são: scale factor, min_neighbors, scale_image e min_size
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray_img, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20, 20))

	for (x, y, width, height) in faces:
		cv2.rectangle(img, (x, y), (x + width, y + height), (255, 0, 0), 2)

		# roi stands for "region of interest"
		roi_gray = gray_img[y:y + height, x: x + width]
		roi_colorful = img[y:y + height, x: x + width]

		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (eye_x, eye_y, eye_width, eye_height) in eyes:
			cv2.rectangle(roi_colorful, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)

	resized_img = ResizeWithAspectRatio(img, width=500)
	cv2.imshow("Image", resized_img)
	cv2.waitKey(0)


if __name__ == '__main__':
	load_image()
