import cv2
import os
import numpy as np
import tensorflow as tf
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Dropout, Flatten, Dense
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


def main():
	casos = cfg.CROPPED + cfg.CASOS + cfg.DSCN_MASK
	controles = cfg.CROPPED + cfg.CONTROLES + cfg.DSCN_MASK

	X, y = loadData([casos, controles])

	vgg_model = VGGFace(include_top=False, input_shape=(584, 584, 3), pooling='max')
	last_layer = vgg_model.get_layer('pool5').output
	x = Dropout(.2, trainable=False, name='custom_dropout1')(last_layer)
	x = Flatten(name='flatten')(x)
	x = Dense(16, activation='relu', name='custom_fc1')(x)
	x = Dropout(.2, trainable=False, name='custom_dropout_2')(x)
	x = Dense(16, activation='relu', name='custom_fc2')(x)
	out = Dense(16, activation='relu', name='custom_fc3')(x)
	custom_vgg_model = Model(vgg_model.input, out)

	custom_vgg_model.compile(optimizer='', loss='', metrics=['accuracy'])

	model.fit(X, y, epochs=10)

	test_loss, test_acc = model.evaluate(X, y, verbose=2)

	print('\nTest accuracy:', test_acc)


if __name__ == '__main__':
	main()
