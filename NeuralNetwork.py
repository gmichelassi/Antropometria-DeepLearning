import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Dropout, Flatten, Dense
from glob import glob
from config import config as cfg
from config import constants as cte


def mean(metric):
	metric_sum = 0
	for i in metric:
		metric_sum = metric_sum + i

	return metric_sum/len(metric)


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

	n_splits = 10
	cv = StratifiedKFold(n_splits=n_splits, random_state=707878)
	for train_index, test_index in cv.split(X, y):
		X_train, y_train = X[train_index], y[train_index]
		X_test, y_test = X[test_index], y[test_index]

		vgg_model = VGGFace(include_top=False, input_shape=(584, 584, 3), pooling='max')
		last_layer = vgg_model.get_layer('pool5').output
		x = Dropout(.2, trainable=False, name='custom_dropout1')(last_layer)
		x = Flatten(name='flatten')(x)
		x = Dense(16, activation='relu', name='custom_fc1')(x)
		x = Dropout(.2, trainable=False, name='custom_dropout_2')(x)
		x = Dense(16, activation='relu', name='custom_fc2')(x)
		out = Dense(16, activation='relu', name='custom_fc3')(x)
		custom_vgg_model = Model(vgg_model.input, out)

		loss, accuracy = [], []
		try:
			# Compilar o modelo escolhido
			custom_vgg_model.compile(optimizer='', loss='', metrics=['accuracy'])

			# Treinar o modelo
			custom_vgg_model.fit(X_train, y_train, epochs=10)

			# Fazer a avaliação do modelo de acordo com as métricas definidas
			test_loss, test_acc = custom_vgg_model.evaluate(X_test, y_test, verbose=2)

			loss.append(test_loss)
			accuracy.append(test_acc)
		except ValueError as ve:
			pass
		except RuntimeError as re:
			pass

		if len(loss) == n_splits and len(accuracy) == n_splits:
			mean_accuracy = mean(accuracy)
			mean_loss = mean(loss)

			# Salvar resultados?
		else:
			pass


if __name__ == '__main__':
	main()
