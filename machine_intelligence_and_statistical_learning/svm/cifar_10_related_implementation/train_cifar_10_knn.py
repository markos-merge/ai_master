import sys
import os

# Add the parent directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cuml import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from open_file import load_CIFAR10, load_CIFAR10_testdata, images_to_numpy
from basic_image_processing import to_hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


def trainHyperKnn():
	print ("Loading CIFAR-10 data...")
	data_folder = './cifar-10'
	cifar10_data = load_CIFAR10( data_folder )
	x_data, y_data = images_to_numpy( cifar10_data, to_gray=True )

	params_grid = {
		'n_neighbors': np.arange(1, 20),
		'weights': ['uniform', 'distance']
	}

	x_hog = to_hog( x_data )

	x_train, x_test, y_train, y_test = train_test_split( x_hog, y_data, test_size=0.2, random_state=42 )

	knn = KNeighborsClassifier()

	grid_search = GridSearchCV( knn, params_grid, cv = 5, verbose = 2, n_jobs = 1 )
	grid_search.fit( x_train, y_train )

	print("Best parameters found by GridSearchCV:")
	print(grid_search.best_params_)
	print(f"GridSearchCV Best Score (training set): {grid_search.best_score_:.2f}%")

	best_knn = grid_search.best_estimator_
	accuracy = best_knn.score( x_test, y_test )
	print(f"Test Accuracy with best estimator: {accuracy:.2f}%")

	return best_knn

def testBestKnn( knn ):
	print ("Loading CIFAR-10 data...")
	data_folder = './cifar-10'
	cifar10_test = load_CIFAR10_testdata( data_folder )
	x_test, y_test = images_to_numpy( cifar10_test, to_gray=True )

	x_hog = to_hog( x_test )
	accuracy = knn.score( x_hog, y_test )
	print(f"Test Accuracy with best estimator: {accuracy:.2f}%")
	class_report = classification_report( y_test, knn.predict( x_hog ) )

	with open( "test_results_knn.txt", "w" ) as f:
		f.write( class_report )


def run():
	knn = trainHyperKnn()
	testBestKnn( knn )

if __name__ == "__main__":
	run()