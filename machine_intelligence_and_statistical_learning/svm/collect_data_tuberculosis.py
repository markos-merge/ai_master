from skimage.restoration import denoise_wavelet
from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu
from skimage.exposure import equalize_hist
from skimage.measure import find_contours
import numpy as np
import os
from tqdm import tqdm

def denoise_image( image ):
	return denoise_wavelet(image, method='BayesShrink', mode='soft', rescale_sigma = True )

def resample_image( image, target_shape ):
	return resize( image, target_shape, order = 3, mode = 'reflect', anti_aliasing = True, preserve_range = True )

def histogram_equilization( image ):
	return equalize_hist( image )

def to_grayscale( image ):
	if image.ndim == 3:
		return rgb2gray(image)
	return img_as_float(image)

def crop_to_object(image):
	thresh = threshold_otsu(image)
	binary_mask = image < thresh

	# Find contours of all objects in the mask
	contours = find_contours(binary_mask, 0.8)

	if not contours:
		return image # Return original if no contours found

	# Find the largest contour by area
	largest_contour = max(contours, key=lambda contour: contour.shape[0])

	# Get the bounding box coordinates of the largest contour
	min_row, min_col = np.min(largest_contour, axis=0)
	max_row, max_col = np.max(largest_contour, axis=0)

	return image[int(min_row):int(max_row), int(min_col):int(max_col)]

def read_images_filenames( folder, file_type = ".png" ):
	filepaths = []
	for filename in os.listdir(folder):
		if filename.lower().endswith( file_type ):
			filepaths.append( os.path.join( folder, filename ) )
	return filepaths

def read_images( folder, file_type = ".png", preprocessing = None, output_fun = None ):
	filepaths = read_images_filenames( folder, file_type )
	print(f"Found {len(filepaths)} images to process in '{folder}'...")

	for image_path in tqdm(filepaths, desc=f"Processing {os.path.basename(folder)}"):
		try:
			image = imread( image_path )

			if preprocessing:
				image = preprocessing( image )

			if output_fun:
				image_filename = os.path.basename(image_path)
				output_fun( image, image_filename )
		except Exception as e:
			print( f"Could not process file: {image_path}. Error: {e}")

def process_image( image ):
	image = to_grayscale( image )
	image = crop_to_object( image )
	image = denoise_image( image )
	image = histogram_equilization( image )
	image = resample_image( image, ( 128, 128 ) )

	return image

class Outputter:
	def __init__( self, output_folder ):
		self.output_folder = output_folder
		os.makedirs( output_folder, exist_ok = True )

	def strip_file_ending( self, filename ):
		return os.path.splitext(filename)[0]

	def __call__( self, image, image_filename ):
		out_name = self.strip_file_ending( image_filename ) + ".png"
		out_name = os.path.join( self.output_folder, out_name )
		try:
			imsave( out_name, img_as_ubyte(image) )
		except Exception as e:
			print( f"Could not save file: {out_name}. Error: {e}" )
			#we must ensure the out_name is removed, in order not to generate corrupt images
			os.remove( out_name )
			raise e


def run():
	outputter = Outputter( "./denoised_images")
	read_images( "tuburculosis/TB_Chest_Radiography_Database/Normal", preprocessing = process_image, output_fun = outputter )
	read_images( "tuburculosis/TB_Chest_Radiography_Database/Tuberculosis", preprocessing = process_image, output_fun = outputter )


if __name__ == "__main__":
	run()