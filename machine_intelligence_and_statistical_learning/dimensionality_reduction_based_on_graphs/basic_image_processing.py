from PIL import Image
import numpy as np
from skimage.feature import hog
import pywt
from hilbertcurve.hilbertcurve import HilbertCurve
from joblib import Parallel, delayed
from tqdm import tqdm

def show_image(image_data, width=32, height=32, is_grayscale=True):
	if is_grayscale:
		img_array = image_data.reshape(height, width)

		if np.issubdtype(img_array.dtype, np.floating):
			img_array = (img_array * 255).astype(np.uint8)
		img = Image.fromarray(img_array, 'L') # 'L' mode is for grayscale
	else:
		# Adjusted to match open_file.py output: (Height, Width, Channels)
		img_array = image_data.reshape(height, width, 3)
		img = Image.fromarray(img_array, 'RGB')

	img.show()

def save_image( image_data, filename, width=32, height=32, is_grayscale=True ):
	if is_grayscale:
		img_array = image_data.reshape(height, width)
		if np.issubdtype(img_array.dtype, np.floating):
			img_array = (img_array * 255).astype(np.uint8)

		img = Image.fromarray(img_array, 'L')
	else:
		# Adjusted to match open_file.py output: (Height, Width, Channels)
		img_array = image_data.reshape(height, width, 3)
		img = Image.fromarray(img_array, 'RGB')

	img.save(filename)

def convertToGrayscale( img_array, to_float = True, width=32, height=32 ):
	ret = np.zeros( height*width, dtype=np.float32 )
	color_step = width*height
	ret = img_array[0:color_step]*0.299 + img_array[color_step:2*color_step]*0.587 + img_array[2*color_step:3*color_step]*0.114

	if not to_float:
		ret = ret.astype( np.uint8 )
	else:
		ret /= 255.
	
	return ret

def extract_hog_features( image_array, pixels_per_cell=(8, 8), cells_per_block=(2, 2), img_size = (32, 32), is_grayscale = False, include_intensity_image = False ):
	"""
	Extracts HOG features from a single grayscale image array.
	The image_array is expected to be 1D, so it will be reshaped.
	"""
	if is_grayscale:
		image = image_array.reshape( img_size[0], img_size[1] )
		hog_features = hog(image, pixels_per_cell=pixels_per_cell,
							cells_per_block=cells_per_block,
							visualize=False,
							feature_vector=True)
		if include_intensity_image:
			hog_features = np.concatenate( ( hog_features, image.flatten() ) )

		return hog_features
	if not is_grayscale:
		image = image_array.reshape( img_size[0], img_size[1], 3 )
		hog_features = hog( image[ :, :, 0 ], pixels_per_cell=pixels_per_cell,
							cells_per_block=cells_per_block,
							visualize=False,
							feature_vector=True)
		
		ret = hog_features

		hog_features = hog( image[ :, :, 1 ], pixels_per_cell=pixels_per_cell,
							cells_per_block=cells_per_block,
							visualize=False,
							feature_vector=True)
		
		ret = np.concatenate( ( ret, hog_features ) )

		hog_features = hog( image[ :, :, 2 ], pixels_per_cell=pixels_per_cell,
							cells_per_block=cells_per_block,
							visualize=False,
							feature_vector=True )
		
		ret = np.concatenate( ( ret, hog_features ) )

		if include_intensity_image:
			gray = image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114
			ret = np.concatenate( ( ret, gray.flatten() ) )


		return ret

def extract_hog_features_wrapper( i, image_array, pixels_per_cell=(4, 4), cells_per_block=(2, 2), img_size = (32, 32), is_grayscale = False, include_intensity_image = False ):
	return ( i, extract_hog_features( image_array, pixels_per_cell, cells_per_block, img_size, is_grayscale, include_intensity_image ) )

def to_hog( images, pixels_per_cell = ( 4, 4 ), cells_per_block = ( 2, 2 ), img_size = ( 32, 32 ), is_grayscale = False, include_intensity_image = False):
	hog_feature_length = len( extract_hog_features( images[0], pixels_per_cell = pixels_per_cell, cells_per_block = cells_per_block, img_size = img_size, is_grayscale = is_grayscale ) )
	ret = np.zeros( ( images.shape[0], hog_feature_length ), dtype=np.float32 )
	tasks = []
	for i in range( images.shape[0] ):
		task = delayed(extract_hog_features_wrapper)( i, images[i], pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, img_size=img_size, is_grayscale=is_grayscale )
		tasks.append( task )

	results = Parallel( n_jobs=-1 )( tqdm( tasks, desc="Extracting HOG Features" ) )
	for i, hog_features in results:
		ret[i] = hog_features

	return ret