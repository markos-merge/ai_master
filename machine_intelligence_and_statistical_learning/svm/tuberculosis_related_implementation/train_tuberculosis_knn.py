import os
import numpy as np
from skimage.io import imread
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier # Import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
		try:
			image = imread( file_path, as_gray = True )
			features = extract_hog_features(image)
			ret_filenames.append( im_name )
			images_list.append( features )
		except Exception as e:
			print(f"Could not process image {file_path}: {e}")

	return ( ret_filenames, np.array( images_list, dtype=np.float32 ) )

def prepareModel():
	"""
	Prepares the data, trains a KNN model using GridSearchCV, and evaluates it.
	"""
	script_dir = os.path.dirname(os.path.abspath(__file__))
	denoised_images_path = os.path.join(os.path.dirname(script_dir), "denoised_images")

	filenames, x_data = import_images( denoised_images_path )

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

	print(f"Original training set size: {len(x_train)}")
	print(f"Original distribution: Normal={np.sum(y_train == -1)}, Tuberculosis={np.sum(y_train == 1)}")

	param_grid = {
		'n_neighbors': [3, 5, 7, 9, 11],
		'weights': ['uniform', 'distance'],
		'metric': ['euclidean', 'manhattan']
	}

	def weighted_recall_scorer(y_true, y_pred):
		recalls_per_class = recall_score(y_true, y_pred, average=None, labels=[-1, 1], zero_division=0)
		weighted_recall = (recalls_per_class[0] * 0.2) + (recalls_per_class[1] * 0.8)
		return weighted_recall
	
	custom_scorer = make_scorer(weighted_recall_scorer, greater_is_better=True)
	
	# Initialize KNeighborsClassifier
	knn_classifier = KNeighborsClassifier()

	print("Starting GridSearchCV to find the best KNN parameters...")
	grid_search = GridSearchCV( knn_classifier, param_grid, cv = 3, verbose = 2, scoring=custom_scorer, n_jobs=-1 ) # n_jobs=-1 for parallel processing
	grid_search.fit(x_train, y_train)

	knn_model = grid_search.best_estimator_
	
	print(f"Model saved to tuberculosis_knn_model.joblib")

	print(f"\nBest parameters: {grid_search.best_params_}")

	print("\n--- Evaluating model on the test set ---")
	y_pred = knn_model.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy on test set: {accuracy:.4f}")
	print("Classification Report:")
	print(classification_report(y_test, y_pred, target_names=['Normal', 'Tuberculosis']))
	print("Confusion Matrix:")
	cm = confusion_matrix(y_test, y_pred)
	print( cm )

	return ( knn_model, x_test, y_test )

def run():
	prepareModel()

if __name__ == "__main__":
	run()
