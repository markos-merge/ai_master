import os
import pickle
import numpy as np
import basic_image_processing as bip

def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def load_CIFAR10( folder ):
	files = os.listdir(folder)

	all_data = []

	for file in files:
		if ( not file.endswith( '.meta') ) & file.startswith( 'data_batch'):
			print( file )
			data_dict = unpickle( os.path.join( folder, file ) )
			all_data.append( data_dict )

	return all_data

def load_CIFAR10_testdata( folder ):
	test_dict = unpickle( os.path.join( folder, 'test_batch' ) )
	
	#We return a list to not have conversions
	return [test_dict]

def load_label_names( folder ):
	meta_dict = unpickle( os.path.join( folder, 'batches.meta' ) )
	label_names = meta_dict[b'label_names']
	return label_names

def get_num_images( images ):
	batches = len( images )
	num_images = 0
	for i in range( batches ):
		num_images += images[i][b'data'].shape[0]
	
	return num_images

def images_to_numpy( images, to_gray = False ):
	batches = len( images )
	num_images = get_num_images( images )
	image_size = len( images[0][b'data'][0] )
	if to_gray:
		image_size = image_size//3

	X = np.zeros( ( num_images, image_size ), dtype=np.float32 )
	Y = np.zeros( ( num_images ), dtype = np.uint8 ) 

	cnt = 0
	for i in range( batches ):
		for j in range( images[i][b'data'].shape[0] ):
			if to_gray:
				img = bip.convertToGrayscale( images[i][b'data'][j], to_float = True, width = 32, height = 32 )
			else:
				img = images[i][b'data'][j]/255.
			X[cnt] = img
			Y[cnt] = images[i][b'labels'][j]
			cnt += 1

	return X, Y




if __name__ == "__main__":
	print ("Loading CIFAR-10 data...")
	data_folder = './cifar-10'
	cifar10_data = load_CIFAR10( data_folder )
	print( f"Loaded {len(cifar10_data)} data batches." )
	X, Y = images_to_numpy( cifar10_data )

	print( f"Converted to numpy arrays: X shape = {X.shape}, Y shape = {Y.shape}" )
	# print( cifar10_data[0][b'labels'])  # Print the contents of the first data batch