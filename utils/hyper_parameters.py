from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization


def get_optimizers():
	return [
		{'opt': SGD(),  'params': 'name=SGD,  lr=default, m=default'},
		{'opt': Adam(), 'params': 'name=Adam, lr=default'}
	]


def get_loss_function():
	return ['binary_crossentropy']


def get_epochs():
	return [10]


def get_layers_ref():
	return [
		'oneDenseFullyConnectedSigmoid'
	]


def get_layers(architeture, last_layers):
	if architeture == 'ZhangFacilRecognitionArchiteture':
		x = Dropout(.2, trainable=False, name='custom_dropout1')(last_layers)
		x = Flatten(name='flatten')(x)
		x = Dense(32, activation='relu', name='custom_fc1')(x)
		x = Dropout(.5, trainable=False, name='custom_dropout_2')(x)
		x = Dense(32, activation='relu', name='custom_fc2')(x)
		out = Dense(1, activation='softmax', name='custom_fc3')(x)
		return out
	elif architeture == 'SinghRareDiseasesArchiteture':
		x = Dense(32, activation='relu', name='custom_fc1')(last_layers)
		x = BatchNormalization()(x)
		x = Dense(16, activation='relu', name='custom_fc2')(x)
		x = BatchNormalization()(x)
		x = Dense(16, activation='relu', name='custom_fc3')(x)
		x = BatchNormalization()(x)
		out = Dropout(.5, trainable=False, name='custom_dropout_1')(x)
		return out
	elif architeture == 'oneDenseFullyConnectedSigmoid':
		out = Dense(1, activation='sigmoid', name='custom_fc1')(last_layers)
		return out
	else:
		raise ValueError('It was not possible to find {0} architeture'.format(architeture))
