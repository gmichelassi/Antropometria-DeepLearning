import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Dropout, Flatten, Dense
from glob import glob
from utils.classifier import Classifier
from config import config as cfg
from config import constants as cte
from config import logger

log = logger.getLogger(__file__)


def buildNNArchiteture(layers, last_layer):
	x = last_layer
	for layer in layers:
		x = (layer)(x)

	return x


def mean(metric):
	metric_sum = 0
	for i in metric:
		metric_sum = metric_sum + i

	return metric_sum/len(metric)


def loadData(img_folder):
	images = []
	for img_path in img_folder:
		log.info("Loading {0} images".format(img_path))
		images = images + glob(img_path)

	X, y = [], []
	for image in images:
		img_absolute_path, img_name = os.path.split(image)
		path, label 				= os.path.split(img_absolute_path)

		X.append(cv2.imread(image))
		y.append(cte.LABELS[label])

	return np.array(X, dtype=object), np.array(y, dtype=np.int)


def main():
	casos = cfg.CROPPED + cfg.CASOS + cfg.DSCN_MASK
	controles = cfg.CROPPED + cfg.CONTROLES + cfg.DSCN_MASK

	X, y = loadData([casos, controles])

	classifier = Classifier()
	params = classifier.getParams()

	for layers in params['layers']:
		for epochs in params['epochs']:
			for losses in params['losses']:
				for metrics in params['metrics']:
					for optimizer in params['optimizers']:

						k = 0
						n_splits = 10
						cv = StratifiedKFold(n_splits=n_splits)
						log.info("Running cross validation k={0}".format(n_splits))

						for train_index, test_index in cv.split(X, y):
							X_train, y_train 	= X[train_index], y[train_index]
							X_test, y_test 		= X[test_index], y[test_index]

							print(type(X_train))
							print(type(X_train[0]))
							print(type(X_train[0][0]))
							print(type(X_train[0][0][0]))

							log.info("#{0} - Building Neural Network architecture".format(k))
							vgg_model = VGGFace(include_top=False, input_shape=(584, 584, 3), pooling='max')
							last_layer = vgg_model.get_layer('pool5').output

							out = buildNNArchiteture(layers, last_layer)

							custom_vgg_model = Model(vgg_model.input, out)

							loss, accuracy = [], []
							try:
								log.info("#{0} - Compiling built model...".format(k))
								custom_vgg_model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

								log.info("#{0} - Training model...".format(k))
								custom_vgg_model.fit(X_train, y_train, epochs=epochs)

								log.info("#{0} - Evaluating model...".format(k))
								test_loss, test_acc = custom_vgg_model.evaluate(X_test, y_test, verbose=2)

								loss.append(test_loss)
								accuracy.append(test_acc)
							except ValueError as ve:
								log.info('[ValueError] Could not perform train and test beacause of error {0}'.format(ve))
							except RuntimeError as re:
								log.info('[RuntimeError] Could not perform train and test beacause of error {0}'.format(re))

							if len(loss) == n_splits and len(accuracy) == n_splits:
								mean_accuracy = mean(accuracy)
								mean_loss = mean(loss)

								log.info("#{0} - Mean accuracy achieved: {1}".format(k, mean_accuracy))
								log.info("#{0} - Mean loss: {1}".format(k, mean_loss))
							else:
								log.info("#{0} - Something went wrong when computing metrics and loss".format(k))

							k = k + 1
							break


if __name__ == '__main__':
	main()
