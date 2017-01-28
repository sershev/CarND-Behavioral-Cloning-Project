from scipy import ndimage
from math import floor
import numpy as np
import pandas as pd
import preprocess_image

import random

def filter_data_frame(csv_file="driving_log.csv", keep_prob=0.5, angle_threshold=0.03):
	df = pd.read_csv(csv_file, header=None)
	for row_idx, row in df.iterrows():
		fname = row[0]
		steering_angle = row[3]
		# skip some 0 angles
		if (abs(steering_angle) < angle_threshold):
			rand = random.randrange(10)
			if (rand < (keep_prob * 10)):
				df.drop(df[row_idx], inplace=True)
	return df

#_______________Import Data______________


def generate_arrays_from_file(csv_file="driving_log.csv", batch_size=128):

	df = pd.read_csv(csv_file, header=None)

	#df = filter_data_frame(df)
	lim = floor(len(df.index)/batch_size) * batch_size

	df.reindex(np.random.permutation(df.index)) # at this place I spent 2 days
	while 1:
		for i in range(0, len(df.index), batch_size):
			batch = df.ix[i:i+batch_size]
			images = []
			angles = []
			for index, row in batch.iterrows():
				fname = row[0]
				steering_angle = row[3]

				# skip some 0 angles
				if (abs(steering_angle) == 0.0):
					rand = random.randrange(10)
					if (rand <5):
					#print('skip: {} - {}'.format(index, fname))
						continue

				# read data
				image = ndimage.imread(fname, mode="RGB")
				image = preprocess_image.preprocess(image)

				# mirror data
				#image2 = np.fliplr(image)
				#steering_angle2 = (-1.0)*steering_angle
#				if (abs(steering_angle) > 0.5):
#					for j in range(0,10):
#				
#						angles.append(steering_angle2)
#						images.append(image2)
#
#						images.append(image)
#						angles.append(steering_angle)
				
				
				#angles.append(steering_angle2)
				#images.append(image2)

				# put data into result array
				images.append(image)
				angles.append(steering_angle)

			features = np.array(images)
			labels = np.array(angles)
			yield (features, labels)


def get_all_data(csv_file="driving_log.csv"):
	df = pd.read_csv(csv_file, header=None)
	images = []
	angles = []
	for index, row in df.iterrows():
		fname = row[0]
		steering_angle = row[3]
		# skip some 0 angles
		#if (abs(steering_angle) < 0.03):
		#	rand = random.randrange(10)
		#	if (rand <9):
				#print('skip: {} - {}'.format(index, fname))
		#		continue

		# read data
		image = ndimage.imread(fname, mode="RGB")
		image = preprocess_image.preprocess(image)
		
		# mirror data
		#if (steering_angle != 0):
		#	image2 = np.fliplr(image)
		#	steering_angle2 = (-1.0)*steering_angle
		#	angles.append(steering_angle2)
		#	images.append(image2)

		# put data into result array
		images.append(image)
		angles.append(steering_angle)

	features = np.array(images)
	labels = np.array(angles)
	return features, labels
