from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from keras.layers import Dropout, Flatten, Dense, BatchNormalization


class Classifier:
	def __init__(self):
		self.optimizers = [
			{'optimizer': SGD(learning_rate=0.001, 	momentum=0.0), 	'params': 'lr=0.001, m=0.0'},
			{'optimizer': SGD(learning_rate=0.01, 	momentum=0.0), 	'params': 'lr=0.01, m=0.0'},
			{'optimizer': SGD(learning_rate=0.1, 	momentum=0.0), 	'params': 'lr=0.1, m=0.0'},
			{'optimizer': SGD(learning_rate=1, 		momentum=0.0), 	'params': 'lr=1, m=0.0'},
			{'optimizer': SGD(learning_rate=10, 	momentum=0.0), 	'params': 'lr=10, m=0.0'},
			{'optimizer': SGD(learning_rate=0.001, 	momentum=0.1), 	'params': 'lr=0.001, m=0.1'},
			{'optimizer': SGD(learning_rate=0.01, 	momentum=0.1), 	'params': 'lr=0.01, m=0.1'},
			{'optimizer': SGD(learning_rate=0.1, 	momentum=0.1), 	'params': 'lr=0.1, m=0.1'},
			{'optimizer': SGD(learning_rate=1, 		momentum=0.1), 	'params': 'lr=1, m=0.1'},
			{'optimizer': SGD(learning_rate=10, 	momentum=0.1), 	'params': 'lr=10, m=0.1'},
			{'optimizer': Adam(learning_rate=0.0001), 				'params': 'lr=0.0001'},
			{'optimizer': Adam(learning_rate=0.001), 				'params': 'lr=0.001'},
			{'optimizer': Adam(learning_rate=0.01), 				'params': 'lr=0.01'},
			{'optimizer': Adam(learning_rate=0.1), 					'params': 'lr=0.1'},
			{'optimizer': Adam(learning_rate=0.5), 					'params': 'lr=0.5'},
			{'optimizer': Adam(learning_rate=1), 					'params': 'lr=1'},
			{'optimizer': Adam(learning_rate=10), 					'params': 'lr=10'}
		]
		self.losses = ['binary_crossentropy']
		self.metrics = [Accuracy(), Precision(), Recall(), ]  # AUC()
		self.epochs = [10, 25]
		self.layers = ['ZhangFacilRecognitionArchiteture', 'SinghRareDiseasesArchiteture', ]

	def getParams(self):
		return {
			'optimizers': 	self.optimizers,
			'losses': 		self.losses,
			'metrics':		self.metrics,
			'epochs': 		self.epochs,
			'layers':		self.layers
		}

	def buildArchiteture(self, architeture_ref, last_layer):
		if architeture_ref == self.layers[0]:
			x = Dropout(.2, trainable=False, name='custom_dropout1')(last_layer)
			x = Flatten(name='flatten')(x)
			x = Dense(16, activation='relu', name='custom_fc1')(x)
			x = Dropout(.5, trainable=False, name='custom_dropout_2')(x)
			x = Dense(8, activation='relu', name='custom_fc2')(x)
			out = Dense(1, activation='relu', name='custom_fc3')(x)
			return out
		elif architeture_ref == self.layers[1]:
			x = Dense(32, activation='relu', name='custom_fc1')(last_layer)
			x = BatchNormalization()(x)
			x = Dense(16, activation='relu', name='custom_fc2')(x)
			x = BatchNormalization()(x)
			x = Dense(16, activation='relu', name='custom_fc3')(x)
			x = BatchNormalization()(x)
			out = Dropout(.5, trainable=False, name='custom_dropout_1')(x)
			return out
		else:
			return None
