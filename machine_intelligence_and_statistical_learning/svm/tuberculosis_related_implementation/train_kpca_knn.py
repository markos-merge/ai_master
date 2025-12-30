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
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, recall_score
from skimage.io import imread
import joblib
import cupy as cp
import gc
from basic_image_processing import save_image


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

img_sz = ( 128, 128 )

def extract_hog_features(image):
	features = hog(image, pixels_per_cell=(16, 16),
				   cells_per_block=(1, 1), visualize=False)
	return features

def import_images( folder ):
	images_list = []
	ret_filenames = []
	filenames = os.listdir( folder )

	for im_name in filenames:
		file_path = os.path.join( folder, im_name )
		image = imread( file_path, as_gray = True )
		features = extract_hog_features(image)
		ret_filenames.append( im_name )
		images_list.append( features )

	return ( ret_filenames, np.array( images_list, dtype=np.float32 ) )

def weighted_recall_scorer(y_true, y_pred):
		recalls_per_class = recall_score(y_true, y_pred, average=None, labels=[-1, 1], zero_division=0)
		weighted_recall = (recalls_per_class[0] * 0.2) + (recalls_per_class[1] * 0.8)
		return weighted_recall
	
def trainNTestKPCAKnn():
	filenames, x_data = import_images( "denoised_images" )

	y_data = np.zeros( x_data.shape[0] )
	for i in range( len( filenames ) ):
		name = filenames[i]
		if name.startswith( "Normal" ):
			y_data[i] = -1
		elif name.startswith( "Tuberculosis" ):
			y_data[i] = 1
		else:
			raise Exception( "Unrecognized image type" )

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=42, stratify=y_data)

	param_grid = {
		'kpca__n_components': [100, 400, 600],
		# 'kpca__n_components': [ 200],
		'kpca__gamma': [0.001, 0.01, 0.1],
		# 'kpca__gamma': [ 0.1],
		
		'lda__n_components': [1],
		
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
		
		( 'knn', KNeighborsClassifier() ),
	])

	custom_scorer = make_scorer(weighted_recall_scorer, greater_is_better=True)
	grid_search = GridSearchCV( pipeline, param_grid, cv = 3, verbose=2, scoring=custom_scorer)
	grid_search.fit( x_train, y_train )
	print( "Best parameters found: ", grid_search.best_params_ )
	class_report = classification_report( y_train, grid_search.predict( x_train ), target_names=['Normal', 'Tuberculosis'] )

	with open( "tuborculosis_train_results_kpca_knn.txt", "w" ) as f:
		f.write( class_report )

	joblib.dump( grid_search.best_estimator_, "tuberculosis_best_kpca_knn_model.joblib" )

	grid_search.fit( x_test, y_test )
	class_report = classification_report( y_test, grid_search.predict( x_test ), target_names=['Normal', 'Tuberculosis'] )
	with open( "tuborculosis_test_results_kpca_knn.txt", "w" ) as f:
		f.write( class_report )

def getSomeClassifications( get_correct = False ):
	model = joblib.load( "tuberculosis_best_kpca_knn_model.joblib" )

	filenames, x_data = import_images( "denoised_images" )

	y_data = np.zeros( x_data.shape[0] )
	for i in range( len( filenames ) ):
		name = filenames[i]
		if name.startswith( "Normal" ):
			y_data[i] = -1
		elif name.startswith( "Tuberculosis" ):
			y_data[i] = 1
		else:
			raise Exception( "Unrecognized image type" )

	cnt = 0
	for i in range( x_data.shape[0] ):
		y_predict = model.predict( x_data[i].reshape(1, -1) )
		if get_correct:
			if y_predict == y_data[i]:
				cnt += 1 
				# filename = f"correct_classification_for_label_{int(y_data[i])}_{int(y_predict[0])}.png"
				print( filenames[i] )
				# save_image( x_data[i], os.path.join( "report/assets/tuburculosis_correct_classifications_kpca_lda_knn", filename), is_grayscale=True )
				if cnt > 2:
					break
		else:
			if y_predict != y_data[i]:
				cnt += 1 
				# filename = f"correct_classification_for_label_{int(y_data[i])}_{int(y_predict[0])}.png"
				print( filenames[i] )
				# save_image( x_data[i], os.path.join( "report/assets/tuburculosis_correct_classifications_kpca_lda_knn", filename), is_grayscale=True )
				if cnt > 2:
					break


if __name__ == "__main__":
	# trainNTestKPCAKnn()
	getSomeClassifications()
