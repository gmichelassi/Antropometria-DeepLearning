import cv2
import os
import pathlib
import sys
import csv
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from glob import glob
from utils.classifier import Classifier
from config import config as cfg
from config import constants as cte
from config import logger

log = logger.getLogger(__file__)
default_shape = (584, 584, 3)
fieldnames = ['image_processing', 'classifier', 'optimizer', 'optmizer_params', 'loss', 'epochs', 'layers', 'mean_accuracy', 'mean_precision', 'mean_recall', 'mean_AUC', 'execution_time']
classifier_name = 'Neural Network'
image_processing = 'dlibHOG'


def mean(metric):
	metric_sum = 0
	for i in metric:
		metric_sum = metric_sum + i

	return metric_sum/len(metric)


def loadData(img_folder, expected_shape):
	images = []
	for img_path in img_folder:
		log.info("Loading {0} images".format(img_path))
		images = images + glob(img_path)

	X, y = [], []
	for image in images:
		img_absolute_path, img_name = os.path.split(image)
		path, label 				= os.path.split(img_absolute_path)

		img = cv2.imread(image)
		if img.shape == expected_shape:
			X.append(img)
			y.append(cte.LABELS[label])
		else:
			log.error("Image {0} has a shape of {1} not matching the expected shape of {2}".format(img_name, img.shape, expected_shape))

	return np.array(X), np.array(y)


def main(expected_shape):
	casos = cfg.CROPPED + cfg.CASOS + cfg.DSCN_MASK
	controles = cfg.CROPPED + cfg.CONTROLES + cfg.DSCN_MASK

	X, y = loadData([casos, controles], expected_shape)
	log.info(f'Data has {y.shape} instaces with shape {X.shape}')

	classifier = Classifier()
	params = classifier.getParams()
	metrics = params['metrics']

	log.info('Preparing output file - writing header')
	with open('./output/test_results.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()

	num_of_tests = len(params['layers']) * len(params['epochs']) * len(params['losses']) * len(params['optimizers'])
	current_test = 1
	for layer_ref in params['layers']:
		for epochs in params['epochs']:
			for losses in params['losses']:
				for optimizer in params['optimizers']:
					k = 0
					n_splits = 10
					start_time = time.time()
					cv = StratifiedKFold(n_splits=n_splits)
					log.info("#{0}/{1} - Running cross validation k={2}".format(current_test, num_of_tests, n_splits))

					log.info("#{0}/{1} - Building Neural Network architecture".format(current_test, num_of_tests))
					vgg_model = VGGFace(include_top=False, input_shape=(584, 584, 3), pooling='max')
					last_layer = vgg_model.get_layer('pool5').output

					final_layers = classifier.buildArchiteture(layer_ref, last_layer)

					if final_layers is None:
						log.error("#{0}/{1} - Erro ao computar arquitetura da rede neural".format(current_test, num_of_tests))
						continue

					custom_vgg_model = Model(vgg_model.input, final_layers)

					log.info("#{0}/{1} - Compiling built model...".format(current_test, num_of_tests))
					custom_vgg_model.compile(optimizer=optimizer['optimizer'], loss=losses, metrics=['accuracy', 'precision', 'recall', 'auc'])

					loss, accuracy, precision, recall, auc = [], [], [], [], []
					log.info("#{0}/{1} - Running cross validation".format(current_test, num_of_tests))
					for train_index, test_index in cv.split(X, y):
						X_train, y_train = X[train_index], y[train_index]
						X_test, y_test = X[test_index], y[test_index]

						log.debug(f'Test shapes: X={X_test.shape}, y={y_test.shape}')
						log.debug(f'Train shapes: X={X_train.shape}, y={y_train.shape}')

						X_train, x_test = X_train/255.0, X_test/255.0

						try:
							log.info("k={0} - Training model...".format(k))
							custom_vgg_model.fit(X_train, y_train, epochs=epochs)

							log.info("k={0} - Evaluating model...".format(k))
							results = custom_vgg_model.evaluate(X_test, y_test, verbose=2)

							loss.append(results[0])
							accuracy.append(results[1])
							precision.append(results[2])
							recall.append(results[3])
							auc.append(results[4])

						except ValueError as ve:
							log.info('[ValueError] Could not perform train and test beacause of error {0}'.format(ve))
						except RuntimeError as re:
							log.info('[RuntimeError] Could not perform train and test beacause of error {0}'.format(re))

						k = k + 1

					if len(accuracy) == n_splits:
						mean_loss = mean(loss)
						mean_accuracy = mean(accuracy)
						mean_precision = mean(precision)
						mean_recall = mean(recall)
						mean_AUC = mean(auc)

						log.info("#{0}/{1} - Mean accuracy achieved: {2}".format(current_test, num_of_tests, mean_accuracy))
						log.info("#{0}/{1} - Mean loss: {2}".format(current_test, num_of_tests, mean_loss))

						execution_time = (time.time() - start_time) / 60

						log.info("#{0}/{1} - Saving results...".format(current_test, num_of_tests))
						with open('./output/test_results.csv', 'a') as csvfile:
							writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
							row = {
								'image_processing': image_processing,
								'classifier': classifier_name,
								'optimizer': optimizer['optimizer'].name,
								'optimizer_params': optimizer['params'],
								'loss': losses,
								'epochs': epochs,
								'layers': layer_ref,
								'mean_accuracy': mean_accuracy,
								'mean_loss': mean_loss,
								'mean_precision': mean_precision,
								'mean_recall': mean_recall,
								'mean_AUC': mean_AUC,
								'execution_time': execution_time
							}
							writer.writerow(row)
							log.info("Results saved!")

						log.info("--- Test {0}/{1} execution time: {2} minutes ---".format(current_test, num_of_tests, execution_time))
					else:
						log.error("#{0}/{1} - Something went wrong when computing metrics and loss".format(current_test, num_of_tests))

					current_test = current_test + 1


if __name__ == '__main__':
	args = sys.argv

	if len(args) == 4:
		shape = (int(args[1]), int(args[2]), int(args[3]))
		main(shape)
	else:
		main(default_shape)
