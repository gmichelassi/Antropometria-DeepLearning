from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from keras.layers import Dropout, Flatten, Dense


class Classifier:
	def __init__(self):
		self.optimizers = [
			SGD(learning_rate=0.01, momentum=0.0),
			Adam(learning_rate=0.001)
		]
		self.losses = ['binary_crossentropy']
		self.metrics = [Accuracy(), Precision(), Recall(), AUC()]
		self.epochs = [10, 100, 1000]

		self.layers = [
			[
				Dropout(.2, trainable=False, name='custom_dropout1'),
				Flatten(name='flatten'),
				Dense(16, activation='relu', name='custom_fc1'),
				Dropout(.2, trainable=False, name='custom_dropout_2'),
				Dense(16, activation='relu', name='custom_fc2'),
				Dense(16, activation='relu', name='custom_fc3')
			],
		]

	def getParams(self):
		return {
			'optimizers': 	self.optimizers,
			'losses': 		self.losses,
			'metrics':		self.metrics,
			'epochs': 		self.epochs,
			'layers':		self.layers
		}
