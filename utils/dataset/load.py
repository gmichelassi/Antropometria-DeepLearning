import cv2
import os
import numpy as np

from config.constants import DEFAULT_SHAPE, LABELS
from glob import glob


def load_data(img_folder):
	images = []
	for img_path in img_folder:
		images += glob(img_path)

	x, y, image_name = [], [], []
	for image in images:
		absolute_path, name = os.path.split(image)
		path_head, label = os.path.split(absolute_path)
		img = cv2.imread(image)
		if img.shape == DEFAULT_SHAPE:
			x.append(img)
			y.append(LABELS[label])
			image_name.append(name)
		else:
			raise RuntimeError(f"The image {name} in {absolute_path} did not match default shape: {DEFAULT_SHAPE}")

	return np.array(x), np.array(y), image_name
