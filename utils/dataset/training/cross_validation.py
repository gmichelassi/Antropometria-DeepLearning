import json
import os

from config.constants import N_SPLITS
from sklearn.model_selection import StratifiedKFold


def default_cross_validation(x, y, custom_vgg_face, epochs):
	loss, accuracy, precision, recall, auc = [], [], [], [], []
	k = 0
	cv = StratifiedKFold(n_splits=N_SPLITS)

	for train_index, test_index in cv.split(x, y):
		results = test_current_fold(x, y, train_index, test_index, custom_vgg_face, epochs)

		loss.append(results[0])
		accuracy.append(results[1])
		precision.append(results[2])
		recall.append(results[3])
		auc.append(results[4])

		k += 1

	return loss, accuracy, precision, recall, auc


def PRP2020_cross_validation(x, y, img_names, custom_vgg_face, epochs):
	loss, accuracy, precision, recall, auc = [], [], [], [], []
	k = 0
	img_names = [x.lower() for x in img_names]
	with open('../../../config/cross_val_folds_PRP.json', 'r') as jsonfile:
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

		results = test_current_fold(x, y, train_index, test_index, custom_vgg_face, epochs)

		loss.append(results[0])
		accuracy.append(results[1])
		precision.append(results[2])
		recall.append(results[3])
		auc.append(results[4])

		k += 1
	return loss, accuracy, precision, recall, auc


def test_current_fold(x, y, train_index, test_index, custom_vgg_face, epochs):
	X_train, y_train = x[train_index], y[train_index]
	X_test, y_test = x[test_index], y[test_index]

	custom_vgg_face.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)
	results = custom_vgg_face.evaluate()

	return results
