import cv2
import cmath
import os
import numpy as np
from decimal import Decimal
from glob import glob
from PIL import Image

from config import constants as cte
from config import logger

from utils.image_processing.transformations import resizeWithAspectRatio, findEyeCoordinates

log = logger.getLogger(__file__)
crop_width = 584
crop_height = 584


def processImages(all_images, show_result=False):
	log.info('Loading face and eye detectors')
	face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')
	eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

	for image_path in all_images:
		original_image = cv2.imread(image_path)

		log.info('Loading results paths')
		img_absolute_path, img_name = os.path.split(image_path)
		head, label = os.path.split(img_absolute_path)

		eyes_path = cte.EYES + '/' + label + '/' + img_name
		gradient_path = cte.GRADIENT + '/' + label + '/' + img_name
		rotated_path = cte.ROTATED + '/' + label + '/' + img_name
		cropped_path = cte.CROPPED + '/' + label + '/' + img_name

		try:
			gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

			log.info('Finding face for {0}'.format(img_name))
			faces = face_cascade.detectMultiScale(gray_image, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20, 20))
			log.info("Found {0} faces!".format(len(faces)))

			if len(faces) == 0:
				log.error(f'Found more than one face for {img_name}')
				return

			x, y, width, height = faces[0]
			cv2.rectangle(original_image, (x, y), (x + width, y + height), (255, 0, 0), 2)

			roi_colorful = original_image[y:y + height, x: x + width]

			found_image = Image.open(image_path)
			detected_face = found_image.crop((x, y, x + width, y + height))
			face = np.asarray(detected_face)
			face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

			h, w = face_gray.shape
			h = int(h / 2)
			crop = face_gray.copy()[0:0 + h, 0:0 + w]

			log.info('Finding eye for {0}'.format(img_name))
			eyes = eye_cascade.detectMultiScale(crop)

			eyes_pair = None
			for (eye_x, eye_y, eye_width, eye_height) in eyes:
				cv2.rectangle(roi_colorful, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (0, 255, 0), 2)
				eyes_pair = face_gray[eye_y:eye_y + eye_height, eye_x:eye_x + eye_width]

			canny = cv2.Canny(face_gray, 50, 100)
			kernel = np.ones((3, 3), np.uint8)
			gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)

			h = gradient.shape[0]
			w = gradient.shape[1]
			x1, y1 = findEyeCoordinates(gradient, h, 0, int(w / 2))
			x1, y1 = abs(x1), abs(y1)
			x2, y2 = findEyeCoordinates(gradient, h, int(w / 2), w)
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
			log.error('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, zde))
		except ValueError as ve:
			log.error('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, ve))
		except IOError as ioe:
			log.error('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, ioe))
		except TypeError as te:
			log.error('It was not possible to preprocesses image {0} because of error {1}'.format(img_name, te))


if __name__ == '__main__':
	images_paths = [cte.CASOS, cte.CONTROLES]

	all_images = []
	for path in images_paths:
		all_images += glob(path)

	processImages(all_images)
