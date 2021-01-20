from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC


class Classifier:
	def __init__(self):
		self.optimizers = [
			SGD(learning_rate=0.01, momentum=0.0),
			Adam(learning_rate=0.001)
		]
		self.losses = [binary_crossentropy()]
		self.metrics = [Accuracy(), Precision(), Recall(), AUC()]
		self.epochs = [10, 100, 1000]
		self.activations = []

	def getParams(self):
		return {
			# Compilar
			'optimizers': 	self.optimizers,
			'losses': 		self.losses,
			'metrics':		self.metrics,
			'epochs': 		self.epochs
		}

	def getLayers(self):
		layers = {
			'': self.activations
		}

		return layers
