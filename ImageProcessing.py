import cv2
import math
import cmath
import os
import numpy as np
from decimal import Decimal
from glob import glob
from PIL import Image

from config import config as cfg
from config import logger

from utils.utils import handleError
from utils.imageProcessing import cropImage, resizeWithAspectRatio, rotateImage, findEyeCoordinates

log = logger.getLogger(__file__)


def faceSize(image_folder):
	log.info('Finding images')
	images = []
	for img in image_folder:
		images = images + glob(img)

	log.info('{0} images found'.format(len(images)))

	FacesSizeX, FacesSizeY = [], []

	log.info('Loading face detector')
	face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')

	for image in images:
		loaded_img = cv2.imread(image)
		gray_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)

		log.info('Findind face for image {0}'.format(image))
		faces = face_cascade.detectMultiScale(gray_img, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20, 20))

		if 0 < len(faces) <= 1:
			for (x, y, width, height) in faces:
				cv2.rectangle(loaded_img, (x, y), (x + width, y + height), (255, 0, 0), 2)

				FacesSizeX.append(width)
				FacesSizeY.append(height)
		else:
			log.info('More than one face found for image ' + image)

	log.info('Calculating mean, max, min and std for all faces detected')
	meanFaceSizeX, meanFaceSizeY = np.mean(FacesSizeX), np.mean(FacesSizeY)
	minFaceSizeX, minFaceSizeY = np.min(FacesSizeX), np.min(FacesSizeY)
	maxFaceSizeX, maxFaceSizeY = np.max(FacesSizeX), np.max(FacesSizeY)
	stdFaceSizeX, stdFaceSizeY = np.std(FacesSizeX), np.std(FacesSizeY)

	log.info('Mean face size X: {0} Y: {1}'.format(meanFaceSizeX, meanFaceSizeY))
	log.info('Max face size  X: {0} Y: {1}'.format(maxFaceSizeX, maxFaceSizeY))
	log.info('Min face size  X: {0} Y: {1}'.format(minFaceSizeX, minFaceSizeY))
	log.info('Std face size  X: {0} Y: {1}'.format(stdFaceSizeX, stdFaceSizeY))

	return {
		'mean': (meanFaceSizeX, meanFaceSizeY),
		'min': (minFaceSizeX, minFaceSizeY),
		'max': (maxFaceSizeX, maxFaceSizeY),
		'std': (stdFaceSizeX, stdFaceSizeY)
	}


def preprocessImage(img_folder, crop_width, crop_height, show_result=False):
	log.info('Finding images')
	images = []
	for img_path in img_folder:
		images = images + glob(img_path)

	log.info('{0} images found'.format(len(images)))

	log.info('Loading face and eye detectors')
	face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
	eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

	for image in images:
		# Onde salvar as imagens
		img_absolute_path, img_name = os.path.split(image)
		path, label = os.path.split(img_absolute_path)

		eyes_path = cfg.EYES + '/' + label + '/' + img_name
		gradient_path = cfg.GRADIENT + '/' + label + '/' + img_name
		rotated_path = cfg.ROTATED + '/' + label + '/' + img_name
		cropped_path = cfg.CROPPED + '/' + label + '/' + img_name

		try:
			original_image = cv2.imread(image)
			gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

			log.info('Finding face for {0}'.format(img_name))
			faces = face_cascade.detectMultiScale(gray_image, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20, 20))
			log.info("Found {0} faces!".format(len(faces)))

			if len(faces) == 0:
				return

			x, y, width, height = faces[0]
			cv2.rectangle(original_image, (x, y), (x + width, y + height), (255, 0, 0), 2)

			# roi stands for "region of interest"
			roi_gray = gray_image[y:y + height, x: x + width]
			roi_colorful = original_image[y:y + height, x: x + width]

			# Recortar somente a face
			found_image = Image.open(image)
			detected_face = found_image.crop((x, y, x + width, y + height))
			face = np.asarray(detected_face)
			face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

			# Encontrar os olhos
			log.info('Finding eye for {0}'.format(img_name))
			eyes = eye_cascade.detectMultiScale(face_gray)

			for (eye_x, eye_y, eye_width, eye_height) in eyes:
				cv2.rectangle(roi_colorful, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)
				eyes_pair = face_gray[eye_y:eye_y + eye_height, eye_x:eye_x + eye_width]
				eyes_box = (eye_x, eye_y, eye_x + eye_width, eye_y + eye_height)

			canny = cv2.Canny(eyes_pair, 50, 245)
			kernel = np.ones((3, 3), np.uint8)
			gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)

			# find coordinates of the pupils
			h = gradient.shape[0]
			w = gradient.shape[1]

			x1, y1 = findEyeCoordinates(gradient, h, 0, int(w / 2))  # left eye
			x1, y1 = abs(x1), abs(y1)  # normalizar geometricamente

			x2, y2 = findEyeCoordinates(gradient, h, int(w / 2), w)  # right eye
			x2, y2 = abs(x2), abs(y2)

			dx, dy = abs(x2 - x1), (abs(y2 - y1) * -1)

			log.info('Finding angle to rotate image')
			z = Decimal(dy) / Decimal(dx)
			alpha_complex = cmath.atan(z)
			alpha = cmath.phase(alpha_complex)
			alpha = alpha / 2

			log.info('Angle found {0}'.format(alpha))
			log.info('Rotating image...')

			img_to_rotate = Image.open(image)
			rotated_image = img_to_rotate.rotate(-alpha)
			rotated_image.save(rotated_path)
			image_rotated = cv2.imread(rotated_path)

			log.info('Cropping image around face...')
			croppedImage = cropImage(image_rotated, x, y, width, height, crop_width, crop_height)

			if show_result:
				log.info('Showing result...')
				resized_img = resizeWithAspectRatio(original_image, width=500)
				cv2.imshow("Original Image", resized_img)
				cv2.waitKey(3000)

				resized_img = resizeWithAspectRatio(croppedImage, width=500)
				cv2.imshow("Processed Image", resized_img)
				cv2.waitKey(3000)

				cv2.destroyAllWindows()

			log.info('Preprocessing done for {0}, saving outputs'.format(img_name))
			cv2.imwrite(eyes_path, eyes_pair)
			cv2.imwrite(gradient_path, gradient)
			cv2.imwrite(cropped_path, croppedImage)

		except ZeroDivisionError as zde:
			handleError('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, zde))
		except ValueError as ve:
			handleError('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, ve))
		except IOError as ioe:
			handleError('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, ioe))
		except TypeError as te:
			handleError('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, te))


if __name__ == '__main__':
	casos = cfg.IMG_DIR + cfg.CASOS + cfg.DSCN_MASK
	controles = cfg.IMG_DIR + cfg.CONTROLES + cfg.DSCN_MASK
	a22q11 = cfg.IMG_DIR + cfg.a22q11 + cfg.IMAGE_MASK
	angelman = cfg.IMG_DIR + cfg.ANGELMAN + cfg.IMAGE_MASK
	apert = cfg.IMG_DIR + cfg.APERT + cfg.IMAGE_MASK
	cdl = cfg.IMG_DIR + cfg.CDL + cfg.IMAGE_MASK
	down = cfg.IMG_DIR + cfg.DOWN + cfg.IMAGE_MASK
	fragilex = cfg.IMG_DIR + cfg.FRAGILEX + cfg.IMAGE_MASK
	marfan = cfg.IMG_DIR + cfg.MARFAN + cfg.IMAGE_MASK
	progeria = cfg.IMG_DIR + cfg.PROGERIA + cfg.IMAGE_MASK
	sotos = cfg.IMG_DIR + cfg.SOTOS + cfg.IMAGE_MASK
	treacher = cfg.IMG_DIR + cfg.TREACHER + cfg.IMAGE_MASK
	turner = cfg.IMG_DIR + cfg.TURNER + cfg.IMAGE_MASK
	williams = cfg.IMG_DIR + cfg.WILLIAMS + cfg.IMAGE_MASK

	all_images = [casos, controles, a22q11, angelman, apert, cdl, down, fragilex, marfan, progeria, sotos, treacher, turner, williams]
	casos_controles_images = [casos, controles]

	preprocessImage(casos_controles_images, 584, 584, True)
