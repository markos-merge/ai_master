from PIL import Image
import numpy as np
from skimage.feature import hog
import pywt
from joblib import Parallel, delayed

import cv2
def show_image(image_data, width=32, height=32, is_grayscale=True):
	"""
	Displays an image from a numpy array using PIL.

	Args:
		image_data (np.array): The image data.
		width (int): The width of the image.
		height (int): The height of the image.
		is_grayscale (bool): If True, assumes a flat 1D grayscale array.
							 If False, assumes a flat 1D CIFAR-10 style RGB array.
	"""
	if is_grayscale:
		# Grayscale image is flat (1024,), needs to be (32, 32)
		# Ensure it's in 0-255 uint8 format for display
		img_array = image_data.reshape(height, width)
		# If the data is float (e.g., 0.0 to 1.0), scale it to 0-255 for display
		if np.issubdtype(img_array.dtype, np.floating):
			img_array = (img_array * 255).astype(np.uint8)
		img = Image.fromarray(img_array, 'L') # 'L' mode is for grayscale
	else:
		# Original CIFAR-10 RGB data is flat (3072,) with R, G, B channels concatenated.
		# We need to reshape and stack it to (32, 32, 3)
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

def transform_to_haar_cascades( image_array ):
	image_array = image_array.reshape( ( 32, 32 ) )
	coeffs = pywt.dwt2(image_array, 'haar')
	LL, (LH, HL, HH) = coeffs

	# 3. Compute "Texture Energy" Features for the SVM
	# Instead of raw pixels, we calculate the energy/variance of each band
	features = []

	# Feature 1-4: Mean of each band (Overall brightness/structure)
	features.extend([np.mean(LL), np.mean(LH), np.mean(HL), np.mean(HH)])

	# Feature 5-8: Standard Deviation (How much texture/variation is in that band)
	features.extend([np.std(LL), np.std(LH), np.std(HL), np.std(HH)])

	# Feature 9-11: Energy of the detailed bands (Sum of squared values)
	# This specifically tells the SVM "how much horizontal/vertical edge info exists"
	features.extend([np.sum(LH**2), np.sum(HL**2), np.sum(HH**2)])
	# extend the image_array to include
	image_array = image_array.ravel()
	np.hstack( ( image_array, features ) )

	return image_array

def extract_hog_features(image_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
	"""
	Extracts HOG features from a single grayscale image array.
	The image_array is expected to be 1D, so it will be reshaped.
	"""
	image = image_array.reshape(32, 32)
	hog_features = hog(image, pixels_per_cell=pixels_per_cell,
					   cells_per_block=cells_per_block,
					   visualize=True,
					   feature_vector=False)
	# print( hog_features )
	return hog_features[1].ravel()

def extract_hog_features_wrapper( i, image_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2) ):
	return ( i, extract_hog_features( image_array, pixels_per_cell, cells_per_block ) )
def to_hog( images ):
	"""
	Assumes the image 32 by 32 and it takes as input a numpy array of images and returns the
	transformed hog images
	"""
	# Calculate the expected HOG feature vector length once.
	# For a 32x32 image with 8x8 cells and 2x2 blocks per cell, this is 324.
	hog_feature_length = len(extract_hog_features(images[0]))
	ret = np.zeros((images.shape[0], hog_feature_length), dtype=np.float32)
	tasks = []
	for i in range( images.shape[0] ):
		task = delayed(extract_hog_features_wrapper)( i, images[i] )
		tasks.append( task )


	results = Parallel(n_jobs=-1)(tqdm(tasks, desc="Extracting HOG Features"))
	for i, hog_features in results:
		ret[i] = hog_features

	return ret

def equalize_histogram(image_array, to_float=True):
	"""
	Performs histogram equalization on a grayscale image array using OpenCV.
	The input is expected to be a 1D numpy array.
	"""
	# cv2.equalizeHist requires a 2D, 8-bit single-channel image (0-255).
	# First, ensure the image is in the correct format.
	if np.issubdtype(image_array.dtype, np.floating):
		image_uint8 = (image_array.reshape(32, 32) * 255).astype(np.uint8)
	else:
		image_uint8 = image_array.reshape(32, 32).astype(np.uint8)

	# Apply histogram equalization
	equalized_img = cv2.equalizeHist(image_uint8)

	# Flatten back to 1D and convert back to float if required
	if to_float:
		return (equalized_img.flatten() / 255.0).astype(np.float32)
	else:
		return equalized_img.flatten().astype(np.uint8)
