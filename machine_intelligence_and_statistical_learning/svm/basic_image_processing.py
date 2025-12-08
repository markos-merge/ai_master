from PIL import Image
import numpy as np
from skimage.feature import hog
import pywt
from joblib import Parallel, delayed
from tqdm import tqdm

import cv2
def show_image(image_data, width=32, height=32, is_grayscale=True):
	if is_grayscale:
		img_array = image_data.reshape(height, width)

		if np.issubdtype(img_array.dtype, np.floating):
			img_array = (img_array * 255).astype(np.uint8)
		img = Image.fromarray(img_array, 'L') # 'L' mode is for grayscale
	else:
		img_array = image_data.reshape(3, height, width).transpose(1, 2, 0)
		img = Image.fromarray(img_array, 'RGB')

	img.show()

def convertToGrayscale( img_array, to_float = True, width=32, height=32 ):
	ret = np.zeros( height*width, dtype=np.float32 )
	color_step = width*height
	ret = img_array[0:color_step]*0.299 + img_array[color_step:2*color_step]*0.587 + img_array[2*color_step:3*color_step]*0.114

	if not to_float:
		ret = ret.astype( np.uint8 )
	else:
		ret /= 255.
	
	return ret

def extract_hog_features(image_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
	"""
	Extracts HOG features from a single grayscale image array.
	The image_array is expected to be 1D, so it will be reshaped.
	"""
	image = image_array.reshape(32, 32)
	hog_features = hog(image, pixels_per_cell=pixels_per_cell,
					   cells_per_block=cells_per_block,
					   visualize=False,
					   feature_vector=True)\

	return hog_features

def extract_hog_features_wrapper( i, image_array, pixels_per_cell=(4, 4), cells_per_block=(2, 2) ):
	return ( i, extract_hog_features( image_array, pixels_per_cell, cells_per_block ) )

def to_hog( images, pixels_per_cell=(4, 4), cells_per_block=(2, 2) ):
	hog_feature_length = len(extract_hog_features(images[0], pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block))
	ret = np.zeros((images.shape[0], hog_feature_length), dtype=np.float32)
	tasks = []
	for i in range( images.shape[0] ):
		# Pass the parameters to the wrapper to ensure consistency
		task = delayed(extract_hog_features_wrapper)( i, images[i], pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block )
		tasks.append( task )

	results = Parallel( n_jobs=-1 )( tqdm( tasks, desc="Extracting HOG Features" ) )
	for i, hog_features in results:
		ret[i] = hog_features

	return ret