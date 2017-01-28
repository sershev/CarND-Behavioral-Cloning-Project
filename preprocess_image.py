import cv2

desired_size = (32,32)

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
	image = resize(image, desired_size)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	#image = filter(image)
	image = normalize_data(image)
	return image
