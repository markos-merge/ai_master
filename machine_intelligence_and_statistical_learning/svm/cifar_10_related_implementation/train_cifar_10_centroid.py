import sys
import os

# Add the parent directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.neighbors import NearestCentroid
import numpy as np
from sklearn.model_selection import GridSearchCV
from open_file import load_CIFAR10, load_CIFAR10_testdata, images_to_numpy, load_label_names
from basic_image_processing import to_hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

def trainHyperCentroid():
	"""Trains and returns the best NearestCentroid classifier after hyperparameter tuning."""
	print ("Loading CIFAR-10 data for training...")
	data_folder = './cifar-10'
	cifar10_data = load_CIFAR10( data_folder )
	x_data, y_data = images_to_numpy( cifar10_data, to_gray=True )

	params_grid = {
		'shrink_threshold': np.logspace(0, 3, 10)
	}

	print("Extracting HOG features...")
	x_hog = to_hog( x_data )

	x_train, x_test, y_train, y_test = train_test_split( x_hog, y_data, test_size=0.2, random_state=42 )

	centroid_clf = NearestCentroid()

	print("Running GridSearchCV for NearestCentroid...")
	grid_search = GridSearchCV( centroid_clf, params_grid, cv = 5, verbose = 2, n_jobs = -1 )
	grid_search.fit( x_train, y_train )

	print("\nBest parameters found by GridSearchCV:")
	print(grid_search.best_params_)
	print(f"GridSearchCV Best Score (on validation set): {grid_search.best_score_:.4f}")

	best_centroid = grid_search.best_estimator_
	accuracy = best_centroid.score( x_test, y_test )
	print(f"Validation Accuracy with best estimator: {accuracy:.4f}")

	return best_centroid

def testBestCentroid( centroid_clf ):
	"""Tests the best classifier on the final test set."""
	print ("\nLoading CIFAR-10 test data for final evaluation...")
	data_folder = './cifar-10'
	cifar10_test = load_CIFAR10_testdata( data_folder )
	x_test, y_test = images_to_numpy( cifar10_test, to_gray=True )

	print("Extracting HOG features from test data...")
	x_hog = to_hog( x_test )
	accuracy = centroid_clf.score( x_hog, y_test )
	print(f"Final Test Accuracy with best estimator: {accuracy:.4f}")
	
	class_report = classification_report( y_test, centroid_clf.predict( x_hog ) )
	print("\nClassification Report on Test Set:")
	print(class_report)

	with open( "test_results_centroid.txt", "w" ) as f:
		f.write( class_report )
		print("\nClassification report saved to test_results_centroid.txt")

if __name__ == "__main__":
	best_model = trainHyperCentroid()
	testBestCentroid( best_model )