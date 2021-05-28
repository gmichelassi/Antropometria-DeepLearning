import csv
import time

from config import constants as cte
from config import logger
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from utils.dataset.load import load_data
from utils.dataset.training.cross_validation import PRP2020_cross_validation, default_cross_validation
from utils.metrics import calculate_mean
from utils.neural_network.hyper_parameters import get_optimizers, get_layers, get_epochs, get_loss_function, \
	get_architeture

log = logger.getLogger(__file__)

IMAGE_PROCESSING = 'dlibHOG'
CLASSIFIER = 'Deep Neural Network'
CROSS_VAL_TYPE = 'PRP2020'
FIELDNAMES = [
	'image_processing', 'classifier', 'optimizer_params', 'loss', 'epochs', 'architeture',
	'mean_accuracy', 'mean_loss', 'mean_precision', 'mean_recall', 'mean_AUC', 'execution_time'
]


def run_neural_network(x, y, image_names):
	log.info('### DEEP NEURAL NETWORK ###')
	log.info('Preparing output file - writing header')

	with open('./output/results.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
		writer.writeheader()

	optimizers = get_optimizers()
	epochs = get_epochs()
	loss_functions = get_loss_function()
	architetures = get_architeture()
	current_test = 0
	num_of_tests = len(optimizers) + len(epochs) + len(loss_functions) + len(architetures)

	for architeture in architetures:
		for epoch in epochs:
			for loss_function in loss_functions:
				for optimizer in optimizers:
					start_time = time.time()

					log.info(f"{current_test}/{num_of_tests} - Building Neural Network architecture")
					vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=cte.DEFAULT_SHAPE)

					for vgg_layer in vgg_model.layers[:15]:
						vgg_layer.trainable = False

					last_layers = vgg_model.output
					custom_model = get_layers(architeture=architeture, last_layers=last_layers)
					transfer_model = Model(inputs=vgg_model.input, outputs=custom_model)

					log.info(f"{current_test}/{num_of_tests} - Compiling model")

					transfer_model.compile(
						loss=loss_function, optimizer=optimizer['opt'],
						metrics=[Accuracy(), Precision(), Recall(), AUC()])

					log.info(f"{current_test}/{num_of_tests} - Running {CROSS_VAL_TYPE} cross validation")

					if CROSS_VAL_TYPE == 'default':
						loss, accuracy, precision, recall, auc = default_cross_validation(x, y, transfer_model, epochs)
					elif CROSS_VAL_TYPE == 'PRP2020':
						loss, accuracy, precision, recall, auc = PRP2020_cross_validation(x, y, image_names, transfer_model, epochs)
					else:
						log.error("Não foi detectado qual validação cruzada deve ser executada")
						return

					if True:
						log.info(f"{current_test}/{num_of_tests} - Calculating results")
						mean_loss = calculate_mean(loss)
						mean_accuracy = calculate_mean(accuracy)
						mean_precision = calculate_mean(precision)
						mean_recall = calculate_mean(recall)
						mean_auc = calculate_mean(auc)

						execution_time = (time.time() - start_time) / 60

						log.info(f"{current_test}/{num_of_tests} - Saving results")
						with open('./output/results.csv', 'a') as csvfile:
							writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
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

					current_test = current_test + 1


if __name__ == '__main__':
	images_paths = [cte.CASOS, cte.CONTROLES]

	x, y, image_names = load_data(images_paths)
	run_neural_network(x, y, image_names)
