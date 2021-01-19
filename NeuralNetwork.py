import cv2
import os
from glob import glob
from config import config as cfg
from config import constants as cte


def loadData(img_folder):
	images = []
	for img_path in img_folder:
		images = images + glob(img_path)

	X, y = [], []
	for image in images:
		img_absolute_path, img_name = os.path.split(image)
		path, label 				= os.path.split(img_absolute_path)

		X.append(cv2.imread(image))
		y.append(cte.LABELS[label])

	return X, y


if __name__ == '__main__':
	casos 		= cfg.CROPPED + cfg.CASOS + cfg.DSCN_MASK
	controles 	= cfg.CROPPED + cfg.CONTROLES + cfg.DSCN_MASK

	X, y = loadData([casos, controles])
	print(len(X), len(y))
