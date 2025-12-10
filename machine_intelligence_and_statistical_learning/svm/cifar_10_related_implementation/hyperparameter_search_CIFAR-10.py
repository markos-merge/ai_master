import numpy as np
# from sklearn.svm import SVC
from open_file import load_CIFAR10, images_to_numpy, load_label_names
from basic_image_processing import to_hog
# import svm as svc

from cuml.svm import SVC
from sklearn.model_selection import train_test_split
from cuml.model_selection import GridSearchCV
from basic_databases_manipulation import getSamplesFromLabels, produceShuffledMatrices
# from sklearn.model_selection import GridSearchCV
import cupy as cp
import gc



def trainSvcModelName( label_0, label_1, X, Y, seed = 0, name = None  ):
	print(f"\n--- Training for labels {label_0} vs {label_1} ---")
	x0, _ = getSamplesFromLabels( X, Y, label_0 )
	x1, _ = getSamplesFromLabels( X, Y, label_1 )
	
	x_data = np.vstack( ( x0, x1 ) )
	x_data = to_hog( x_data )
	y_data = np.hstack( (np.ones(len(x0)) * -1, np.ones(len(x1))) )
	print(f"Combined data shape: X={x_data.shape}, Y={y_data.shape}")

	x_shuffled, y_shuffled = produceShuffledMatrices( seed, x_data, y_data )
	# x_shuffled = cp.asarray( x_shuffled )
	# y_shuffled = cp.asarray( y_shuffled )
	print("Data shuffled.")

	parameters = { 'C': np.arange( 0.06, 0.6, 0.02 ), 'gamma': np.arange( 0.01, 1., .2 ) }
	model = GridSearchCV( SVC( kernel = "rbf" ), param_grid = parameters, n_jobs = None, verbose = 2 )
	model.fit( x_shuffled, y_shuffled )
	print("Best parameters found by GridSearchCV:")
	print(model.best_params_)
	print(f"GridSearchCV Best Score (training set): {model.best_score_:.2f}%")
	best_svm = model.best_estimator_

	if name:
		model_filename = f"./results/svm_model_{name}.pkl"
		report_filename = f"./results/svm_report_{name}.txt"
		# Note: Scikit-learn's SVC can be directly pickled
		import pickle
		with open(model_filename, 'wb') as f:
			pickle.dump(best_svm, f)
		print(f"Model saved to {model_filename}")

		with open( report_filename, 'w' ) as f:
			f.write(f"--- SVM Training Report for {name} ---\n")
			f.write(f"Labels Classified: {label_0} vs {label_1}\n")
			f.write(f"Best Parameters found by GridSearchCV: {model.best_params_}\n")
			f.write(f"GridSearchCV Best Score (training set): {model.best_score_:.2f}%\n")
		print(f"Training report saved to {report_filename}")


if __name__ == "__main__":
	print ("Loading CIFAR-10 data...")
	data_folder = './cifar-10'
	cifar10_data = load_CIFAR10( data_folder )
	label_names = [name.decode('utf-8') for name in load_label_names(data_folder)]
	print( f"Loaded {len(cifar10_data)} data batches." )
	X, Y = images_to_numpy( cifar10_data, True )

	print(f"Converted to numpy arrays: X shape = {X.shape}, Y shape = {Y.shape}")

	i_start = 0
	j_start = 1
	for i in range( i_start, 10 ):
		if not i == i_start:
			j_start = i + 1
		for j in range( j_start, 10 ):
			label = str( i ) + "vs" + str( j )
			print(f"Preparing to classify '{label_names[i]}' vs '{label_names[j]}'")
			trainSvcModelName( i, j, X, Y, seed = 42, name = label )
			gc.collect()
			mempool = cp.get_default_memory_pool()
			mempool.free_all_blocks()
			pinned_mempool = cp.get_default_pinned_memory_pool()
			pinned_mempool.free_all_blocks()
