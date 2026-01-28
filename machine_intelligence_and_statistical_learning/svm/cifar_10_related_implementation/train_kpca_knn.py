import sys
import os

# Add the parent directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pca_implementation import KernelPCA
# from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from cuml.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from open_file import load_CIFAR10, images_to_numpy, load_CIFAR10_testdata
from basic_image_processing import to_hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import joblib
import cupy as cp
import gc


class ToNumpy(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		if hasattr(X, 'get'):
			return X.get()
		return X

class GPUCleaner(BaseEstimator, TransformerMixin):
	def fit(self, X, y=None):
		self._clean()
		return self

	def transform(self, X):
		self._clean()
		return X

	def _clean(self):
		gc.collect()
		mempool = cp.get_default_memory_pool()
		mempool.free_all_blocks()
		pinned_mempool = cp.get_default_pinned_memory_pool()
		pinned_mempool.free_all_blocks()

def trainKPCAKnn():
	print ("Loading CIFAR-10 data...")
	data_folder = './cifar-10'
	cifar10_data = load_CIFAR10( data_folder )
	x_data, y_data = images_to_numpy( cifar10_data, to_gray=True )

	x_hog = to_hog( x_data )
	x_hog = x_hog.astype(np.float32)

	print( x_hog.shape )
	param_grid = {
		'kpca__n_components': [200, 400, 600],
		# 'kpca__n_components': [ 200],
		'kpca__gamma': [0.001, 0.01, 0.1],
		# 'kpca__gamma': [ 0.1],
		
		'lda__n_components': [2, 5, 9],
		# 'lda__n_components': [ 9],
		
		'knn__n_neighbors': [3, 5, 7, 9, 11, 13]
		# 'knn__n_neighbors': [11]
	}

	print ("Splitting data into train and test sets...")
	
	pipeline = Pipeline([
		( 'kpca', KernelPCA( kernel='rbf' ) ), 
	
		##utils
		('cleaner', GPUCleaner()),
		( 'to_cpu', ToNumpy() ),

		( 'lda', LinearDiscriminantAnalysis() ),
		
		( 'knn', KNeighborsClassifier( n_jobs = -1 ) ),
	])

	grid_search = GridSearchCV( pipeline, param_grid, cv = 3, verbose=2 )
	grid_search.fit( x_hog, y_data )
	print( "Best parameters found: ", grid_search.best_params_ )
	class_report = classification_report( y_data, grid_search.predict( x_hog ) )

	with open( "cifar_10_train_results_kpca_knn.txt", "w" ) as f:
		f.write( class_report )

	joblib.dump( grid_search.best_estimator_, "cifar_10_best_kpca_knn_model.joblib" )

def testKPCAKnn():
	print ("Loading CIFAR-10 data...")
	data_folder = './cifar-10'
	cifar10_test = load_CIFAR10_testdata( data_folder )
	x_test, y_test = images_to_numpy( cifar10_test, to_gray=True )

	x_hog = to_hog( x_test )
	x_hog = x_hog.astype(np.float32)

	model = joblib.load( "cifar_10_related_implementation/cifar_10_best_kpca_knn_model.joblib")
	print( model )
	accuracy = model.score( x_hog, y_test )
	print(f"Test Accuracy with best estimator: {accuracy:.2f}%")
	class_report = classification_report( y_test, model.predict( x_hog ) )

	with open( "cifar_10_test_results_kpca_knn.txt", "w" ) as f:
		f.write( class_report )

if __name__ == "__main__":
	# trainKPCAKnn()
	testKPCAKnn()