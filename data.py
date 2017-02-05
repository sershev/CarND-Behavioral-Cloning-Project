from scipy import ndimage
from math import floor
import numpy as np
import pandas as pd
import preprocess_image

import random

def generate_arrays_from_file(csv_file="new_driving_log.csv", batch_size=128):

	df = pd.read_csv(csv_file, header=None)

	while 1:
		df.reindex(np.random.permutation(df.index)) 
		for i in range(0, len(df.index), batch_size):
			batch = df.ix[i:i+batch_size]
			images = []
			angles = []
			for index, row in batch.iterrows():
				fname = row[0]
				steering_angle = row[3]

				# skip some 0 angles
				#if (abs(steering_angle) == 0.0):
				#	rand = random.randrange(10)
				#	if (rand <5):
				#	#print('skip: {} - {}'.format(index, fname))
				#		continue

				# read data
				image = ndimage.imread(fname, mode="RGB")
				image = preprocess_image.preprocess(image)

				# put data into result array
				images.append(image)
				angles.append(steering_angle)

				# Data augmentation
				if (abs(steering_angle) != 0.0):
					image2 = np.fliplr(image)
					steering_angle2 = (-1.0)*steering_angle

					images.append(image2)
					angles.append(steering_angle2)

			features = np.array(images)
			labels = np.array(angles)
			yield (features, labels)
