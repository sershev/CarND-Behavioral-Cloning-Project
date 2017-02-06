import cv2
import config as cfg
import numpy as np

#import matplotlib.pyplot as plt

desired_size = (cfg.CONFIG['img_width'], cfg.CONFIG['img_height'])

def resize(img, new_size):
	res = cv2.resize(img, new_size, interpolation = cv2.INTER_CUBIC)
	return res


def gray_scale(img):
	return NotImplemented

def filter(img):
	filtered_image = cv2.Laplacian(img,cv2.CV_32F)
	return filtered_image


def normalize_data(data):
    a = -0.5
    b = 0.5
    min_val = 0
    max_val = 255
    return a + (data - min_val)*(b-a)/(max_val - min_val)

def preprocess(image):
	image = np.delete(image, range(0,(int)(cfg.CONFIG['img_height']/3)+10), axis=0)
	image = resize(image, desired_size)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	#image = filter(image)
	#image = normalize_data(image)
	#plt.imshow(image)
	#plt.show()


	return image
