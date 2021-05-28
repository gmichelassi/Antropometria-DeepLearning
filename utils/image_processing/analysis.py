import cv2
import numpy as np

from glob import glob


def faceSize(image_folder):
    images = []
    for img in image_folder:
        images = images + glob(img)

    faces_size_x, faces_size_y = [], []

    face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_alt2.xml')

    for image in images:
        loaded_img = cv2.imread(image)
        gray_img = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5, cv2.CASCADE_SCALE_IMAGE, (20, 20))

        if 0 < len(faces) <= 1:
            for (x, y, width, height) in faces:
                cv2.rectangle(loaded_img, (x, y), (x + width, y + height), (255, 0, 0), 2)

                faces_size_x.append(width)
                faces_size_y.append(height)
        else:
            return

    mean_face_size_x, mean_face_size_y = np.mean(faces_size_x), np.mean(faces_size_y)
    min_face_size_x, min_face_size_y = np.min(faces_size_x), np.min(faces_size_y)
    max_face_size_x, max_face_size_y = np.max(faces_size_x), np.max(faces_size_y)
    std_face_size_x, std_face_size_y = np.std(faces_size_x), np.std(faces_size_y)

    return {
        'mean': (mean_face_size_x, mean_face_size_y),
        'min': (min_face_size_x, min_face_size_y),
        'max': (max_face_size_x, max_face_size_y),
        'std': (std_face_size_x, std_face_size_y)
    }
