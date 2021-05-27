import cv2
import os
import pathlib
import sys
import csv
import time
import json
import numpy as np
import tensorflow as tf
from utils.utils import calculate_mean
from glob import glob
from config import constants as cte
from config import logger
from tensorflow.keras import Model
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import StratifiedKFold
from utils.hyper_parameters import get_optimizers, get_layers, get_epochs, get_loss_function, get_layers_ref

log = logger.getLogger(__file__)
DEFAULT_SHAPE = (584, 584, 3)
IMAGE_PROCESSING = 'dlibHOG'
CLASSIFIER = 'Deep Neural Network'
CROSS_VAL_TYPE = 'PRP2020'  # 'default'
N_SPLITS = 10
FIELDNAMES = ['image_processing', 'classifier', 'optimizer_params', 'loss', 'epochs', 'architeture',
			  'mean_accuracy', 'mean_loss', 'mean_precision', 'mean_recall', 'mean_AUC', 'execution_time']


def load_data(img_folder):
	images = []
	for img_path in img_folder:
		images += glob(img_path)

	X, y, image_name = [], [], []
	for image in images:
		absolute_path, name = os.path.split(image)
		path_head, label = os.path.split(absolute_path)
		img = cv2.imread(image)
		if img.shape == DEFAULT_SHAPE:
			X.append(img)
			y.append(cte.LABELS[label])
			image_name.append(name)
		else:
			raise RuntimeError(f"The image {name} in {absolute_path} did not match default shape: {DEFAULT_SHAPE}")

	return np.array(X), np.array(y), image_name


def default_cross_validation(X, y, custom_vgg_face, epochs):
	loss, accuracy, precision, recall, auc = [], [], [], [], []
	k = 0
	cv = StratifiedKFold(n_splits=N_SPLITS)

	for train_index, test_index in cv.split(X, y):
		results = test_current_fold(X, y, train_index, test_index, custom_vgg_face, epochs)

		loss.append(results[0])
		accuracy.append(results[1])
		precision.append(results[2])
		recall.append(results[3])
		auc.append(results[4])

		k += 1

	return loss, accuracy, precision, recall, auc


def PRP2020_cross_validation(X, y, img_names, custom_vgg_face, epochs):
	loss, accuracy, precision, recall, auc = [], [], [], [], []
	k = 0
	img_names = [x.lower() for x in img_names]
	with open('./img/cross_val_folds_PRP.json', 'r') as jsonfile:
		folds = json.load(jsonfile)

	for current_fold in folds:
		test_index = []
		train_index = []
		for img_path in folds[current_fold]['test']:
			img_absolute_path, img_name = os.path.split(img_path)
			lower_img_name = img_name.lower()
			if lower_img_name in img_names:
				test_index.append(img_names.index(lower_img_name))

		for img_path in folds[current_fold]['train']:
			img_absolute_path, img_name = os.path.split(img_path)
			lower_img_name = img_name.lower()
			if lower_img_name in img_names:
				train_index.append(img_names.index(lower_img_name))

		log.info(f"Testing fold {k}/{N_SPLITS}")
		results = test_current_fold(X, y, train_index, test_index, custom_vgg_face, epochs)

		loss.append(results[0])
		accuracy.append(results[1])
		precision.append(results[2])
		recall.append(results[3])
		auc.append(results[4])

		k += 1
	return loss, accuracy, precision, recall, auc


def test_current_fold(X, y, train_index, test_index, custom_vgg_face, epochs):
	X_train, y_train = X[train_index], y[train_index]
	X_test, y_test = X[test_index], y[test_index]

	custom_vgg_face.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
	results = custom_vgg_model.evaluate()

	return results


def run_neural_network(X, y, image_names):
	log.info('### DEEP NEURAL NETWORK ###')
	log.info('Preparing output file - writing header')
	with open('./output/results.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
		writer.writeheader()

	optimizers = get_optimizers()
	epochs = get_epochs()
	loss_functions = get_loss_function()
	architetures = get_layers_ref()
	current_test = 0
	num_of_tests = len(optimizers) + len(epochs) + len(loss_functions) + len(layers)

	for architeture in architetures:
		for epoch in epochs:
			for loss_function in loss_functions:
				for optimizer in optimizers:
					start_time = time.time()

					log.info(f"{current_test}/{num_of_tests} - Building Neural Network architecture")
					vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=DEFAULT_SHAPE)

					for vgg_layer in vgg_model.layers[:15]:
						vgg_layer.trainable = False

					last_layers = vgg_model.output
					custom_model = get_layers(architeture=architeture, last_layers=last_layers)
					transfer_model = Model(inputs=vgg_model.input, outputs=custom_model)

					log.info(f"{current_test}/{num_of_tests} - Compiling model")
					transfer_model.compile(loss=loss_function, optimizer=optimizer['opt'],
										   metrics=[Accuracy(), Precision(), Recall(), AUC()])

					log.info(f"{current_test}/{num_of_tests} - Running {CROSS_VAL_TYPE} cross validation")
					if CROSS_VAL_TYPE == 'default':
						loss, accuracy, precision, recall, auc = default_cross_validation(X, y, transfer_model, epochs)
					elif CROSS_VAL_TYPE == 'PRP2020':
						loss, accuracy, precision, recall, auc = PRP2020_cross_validation(X, y, image_names, transfer_model, epochs)
					else:
						log.error("Não foi detectado qual validação cruzada deve ser executada")
						return

					if CONDITION:
						log.info(f"{current_test}/{num_of_tests} - Calculating results")
						mean_loss = calculate_mean(loss)
						mean_accuracy = calculate_mean(accuracy)
						mean_precision = calculate_mean(precision)
						mean_recall = calculate_mean(recall)
						mean_auc = calculate_mean(auc)

						execution_time = (time.time() - start_time) / 60

						log.info(f"{current_test}/{num_of_tests} - Saving results")
						with open('./output/test_results.csv', 'a') as csvfile:
							writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
							row = {
								'image_processing': IMAGE_PROCESSING,
								'classifier': CLASSIFIER,
								'optimizer_params': optimizer['params'],
								'loss': loss_function,
								'epochs': epoch,
								'architeture': architeture,
								'mean_accuracy': mean_accuracy,
								'mean_loss': mean_loss,
								'mean_precision': mean_precision,
								'mean_recall': mean_recall,
								'mean_AUC': mean_auc,
								'execution_time': execution_time
							}
							writer.writerow(row)
					else:
						log.error(f"{current_test}/{num_of_tests} - Não foi possível calcular corretamente as métricas")
						return

					current_test = current_test + 1


if __name__ == '__main__':
	casos = cte.CROPPED + cte.CASOS + cte.DSCN_MASK
	controles = cte.CROPPED + cte.CONTROLES + cte.DSCN_MASK

	X, y, image_names = load_data([casos, controles])
	run_neural_network(X, y, image_names)
