from open_file import load_CIFAR10



if __name__ == "__main__":
	print ("Loading CIFAR-10 data...")
	data_folder = './cifar-10'
	cifar10_data = load_CIFAR10( data_folder )
	print( f"Loaded {len(cifar10_data)} data batches." )
	print( len( cifar10_data[0][b'data'][0] ) )  # Print the contents of the first data batch