from keras.models import Sequential
#from keras.layers import Input
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.activations import relu, elu
from keras.callbacks import ModelCheckpoint

import numpy as np
import data
import config as cfg


#__________________Model_________________
def create_model():
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode="valid", input_shape=(cfg.CONFIG['img_height'],cfg.CONFIG['img_width'],3)))
	model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode="valid", dim_ordering="default"))
	model.add(Activation("elu"))
	
	model.add(Flatten())
	
	model.add(Dense(100))

	model.add(Dense(1))
	model.add(Activation('linear'))
	return model

def save_model(model, filename="model.json"):
	# serialize model to JSON
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)

def save_model_weights(model, filename="model.h5"):
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")

def load_model_weights(model, filename="model.h5"):
	try:
		model.load_weights(filename)
	except OSError as e:
		print(filename + " not found! Continue training without weights!")
	finally:
		return model

model = create_model()
mdoel = load_model_weights(model)

#_______________Training_________________
model.compile('adam', 'mse', ['accuracy'])
#ModelCheckpoint("sweights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.fit_generator(data.generate_arrays_from_file(), samples_per_epoch=37000, nb_epoch=1)

save_model(model)
save_model_weights(model)


