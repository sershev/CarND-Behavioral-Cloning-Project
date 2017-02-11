from keras.models import Sequential
#from keras.layers import Input
from keras.layers.core import Dense, Flatten, Activation, Dropout, Lambda
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
	model.add(Lambda( lambda x: x/127-1, input_shape=(cfg.CONFIG['img_height'],cfg.CONFIG['img_width'], 3)))
	
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), border_mode="valid", init="normal" ))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))

	model.add(Convolution2D(36, 5, 5, subsample=(2,2), border_mode="valid", init="normal"))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))

	model.add(Convolution2D(48, 5, 5, subsample=(2,2), border_mode="valid", init="normal"))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))

	model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", init="normal"))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))

	model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode="valid", init="normal"))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))
	
	model.add(Flatten())
	
	model.add(Dense(1164))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))

	model.add(Dense(100))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))

	model.add(Dense(50))
	model.add(Activation("elu"))
	model.add(Dropout(0.5))

	model.add(Dense(10))
	model.add(Activation("elu"))

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
model.compile('adam', 'mse', ['mse'])
model.fit_generator(data.generate_arrays_from_file(), 
					validation_data=data.generate_arrays_from_file(csv_file=cfg.CONFIG['validate_csv']), 
					nb_val_samples=cfg.CONFIG['nb_val_samples'], 
					samples_per_epoch=cfg.CONFIG['samples_per_epoch'], 
					nb_epoch=cfg.CONFIG['epochs'])

save_model(model)
save_model_weights(model)


