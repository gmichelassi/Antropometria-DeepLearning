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
crop_width = 584
crop_height = 584


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


def rescaleAndProcessImage(images):
	for image_path in images:
		try:
			original_image = cv2.imread(image_path)
			(h, w) = original_image.shape[:2]

			img_absolute_path, img_name = os.path.split(image_path)
			resized_path = img_absolute_path + '/rescaled_' + img_name

			scale_factor = .9
			isImageReady = False
			while not isImageReady:
				resized_image = resizeWithAspectRatio(original_image, int(w * scale_factor))
				cv2.imwrite(resized_path, resized_image)
				processImage(image_path=resized_path, original_image=resized_image, show_result=True)

				r = input('Resultado do processamento foi o esperado (y/n): ')
				if r == 'y':
					isImageReady = True
				else:
					scale_factor = scale_factor * 0.9
		except IOError as ioe:
			handleError("Its was not possible to load image {0} due to error {1}".format(image_path, ioe))


def defaultProcessing(img_folder):
	log.info('Finding images')
	images = []
	for img_path in img_folder:
		images = images + glob(img_path)

	log.info('{0} images found'.format(len(images)))

	for image_path in images:
		original_image = cv2.imread(image_path)
		processImage(image_path=image_path, original_image=original_image, show_result=False)


def processImage(image_path, original_image, show_result=False):
	log.info('Loading face and eye detectors')
	face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
	eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

	img_absolute_path, img_name = os.path.split(image_path)
	path, label = os.path.split(img_absolute_path)

	eyes_path = cfg.EYES + '/' + label + '/' + img_name
	gradient_path = cfg.GRADIENT + '/' + label + '/' + img_name
	rotated_path = cfg.ROTATED + '/' + label + '/' + img_name
	cropped_path = cfg.CROPPED + '/' + label + '/' + img_name

	try:
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
		found_image = Image.open(image_path)
		detected_face = found_image.crop((x, y, x + width, y + height))
		face = np.asarray(detected_face)
		face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

		h, w = face_gray.shape
		h = int(h / 2)
		crop = face_gray.copy()[0:0 + h, 0:0 + w]
		# Encontrar os olhos
		log.info('Finding eye for {0}'.format(img_name))
		eyes = eye_cascade.detectMultiScale(crop)

		eyes_pair = None
		for (eye_x, eye_y, eye_width, eye_height) in eyes:
			cv2.rectangle(roi_colorful, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)
			eyes_pair = face_gray[eye_y:eye_y + eye_height, eye_x:eye_x + eye_width]

		canny = cv2.Canny(face_gray, 50, 245)
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

		img_to_rotate = Image.open(image_path)
		rotated_image = img_to_rotate.rotate(-alpha)
		rotated_image.save(rotated_path)
		image_rotated = cv2.imread(rotated_path)

		log.info('Cropping image around face...')

		gray_rotated_image = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2GRAY)
		rotated_faces = face_cascade.detectMultiScale(gray_rotated_image, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20, 20))
		if len(rotated_faces) == 0:
			log.warning('Houve um erro ao detectar a face na imagem {0} rotacionada. Tentando novamente com menos parâmetros.'.format(img_name))
			rotated_faces2 = face_cascade.detectMultiScale(gray_rotated_image)
			face_x, face_y, face_width, face_height = rotated_faces2[0]
		else:
			face_x, face_y, face_width, face_height = rotated_faces[0]
		rotated_face = image_rotated[face_y:face_y + face_height, face_x:face_x + face_width]
		croppedImage = resizeWithAspectRatio(rotated_face, crop_width, crop_height)

		if show_result:
			log.info('Showing result...')
			resized_original_img = resizeWithAspectRatio(original_image, width=500)
			cv2.imshow("Original Image", resized_original_img)
			cv2.waitKey(3000)

			resized_final_img = resizeWithAspectRatio(croppedImage, width=500)
			cv2.imshow("Processed Image", resized_final_img)
			cv2.waitKey(3000)

			cv2.destroyAllWindows()

		log.info('Preprocessing done for {0}, saving outputs'.format(img_name))
		if eyes_pair is not None:
			cv2.imwrite(eyes_path, eyes_pair)
		else:
			log.warning('Could not save eyes pair for image {0}.')

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

	isTest = True

	if isTest:
		# image_path = cfg.IMG_DIR + '/_testImage/MIT-10.jpg'
		# original_image = cv2.imread(image_path)
		# processImage(image_path=image_path, original_image=original_image, show_result=True)
		erro_eyes_pair = [
			cfg.IMG_DIR + cfg.CASOS + '/DSCN3469.JPG',
		]

		erro_detectar_faces = [
			cfg.IMG_DIR + cfg.CONTROLES + '/DSCN4200.JPG',
			cfg.IMG_DIR + cfg.CONTROLES + '/DSCN4102.JPG'
		]

		erro_zero_division = [
			cfg.IMG_DIR + cfg.CASOS + '/DSCN3683.JPG',
			cfg.IMG_DIR + cfg.CASOS + '/DSCN3603.JPG',
			cfg.IMG_DIR + cfg.CASOS + '/DSCN3421.JPG',
			cfg.IMG_DIR + cfg.CONTROLES + '/DSCN3865.JPG',
			cfg.IMG_DIR + cfg.CONTROLES + '/DSCN3911.JPG']

		for img in erro_zero_division:
			original_image = cv2.imread(img)
			processImage(image_path=img, original_image=original_image, show_result=False)
	else:
		defaultProcessing(casos_controles_images)
