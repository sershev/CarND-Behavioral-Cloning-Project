from keras.models import Sequential
#from keras.layers import Input
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.activations import relu, softmax

import numpy as np
import data


#__________________Model_________________
def create_model():
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode="valid", input_shape=(32, 32, 3)))
	model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode="valid", dim_ordering="default"))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))
	model.add(Flatten(input_shape=(8, 8, 16)))
	model.add(Dense(128))
	model.add(Activation('relu'))
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
model.fit_generator(data.generate_arrays_from_file(), samples_per_epoch=400, nb_epoch=6)

save_model(model)
save_model_weights(model)


