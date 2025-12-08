from cuml import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
import os
import numpy as np
from skimage.io import imread
from skimage.feature import hog
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
		image = imread( file_path, as_gray = True )
		features = extract_hog_features(image)
		ret_filenames.append( im_name )
		images_list.append( features )

	return ( ret_filenames, np.array( images_list, dtype=np.float32 ) )

def prepareModel():
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

	# Split data into training and testing sets BEFORE augmentation
	# This is crucial to prevent augmented data from leaking into the test set
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, stratify=y_data)

	# We are removing the aggressive undersampling.
	# The class_weight parameter in SVC and the custom scorer are better
	# suited to handle the imbalance without discarding data.
	print(f"Original training set size: {len(x_train)}")
	print(f"Original distribution: Normal={np.sum(y_train == -1)}, Tuberculosis={np.sum(y_train == 1)}")

	param_grid = [
		{
			'kernel': ['rbf'],
			'C': np.arange( 0.1, 1., .2 ),
			'gamma': [1., 0.1, 0.01, 0.001, 0.0001, 'scale'] 
		},
		{	'kernel': ['poly'],
			'C': np.arange( 0.1, 1., .2 ),
			'gamma': [1., 0.1, 0.01, 0.001, 'scale'],
			'degree': [3, 5],
			'coef0': [-1.5, -1., .5, 0.01] }
	]

	class_weights = {-1: 0.2, 1: 0.8}
	
	def weighted_recall_scorer(y_true, y_pred):
		recalls_per_class = recall_score(y_true, y_pred, average=None, labels=[-1, 1], zero_division=0)
		weighted_recall = (recalls_per_class[0] * 0.2) + (recalls_per_class[1] * 0.8)
		return weighted_recall
	
	custom_scorer = make_scorer(weighted_recall_scorer, greater_is_better=True)
	
	# Use the custom_scorer to guide the hyperparameter search
	grid_search = GridSearchCV( SVC(class_weight=class_weights, probability=True), param_grid, cv = 3, verbose = 2, scoring=custom_scorer )
	grid_search.fit(x_train, y_train)

	svm_model = grid_search.best_estimator_
	print(f"Best parameters: {grid_search.best_params_}")

	# Save the best model to a file
	model_filename = 'best_svm_model.joblib'
	joblib.dump(svm_model, model_filename)
	print(f"Best model saved to {model_filename}")

	print("\n--- Evaluating model on the test set ---")
	y_pred = svm_model.predict(x_test)
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy on test set: {accuracy:.4f}")
	print("Classification Report:")
	print(classification_report(y_test, y_pred, target_names=['Normal', 'Tuberculosis']))
	print("Confusion Matrix:")
	cm = confusion_matrix(y_test, y_pred)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
				xticklabels=['Normal', 'Tuberculosis'], 
				yticklabels=['Normal', 'Tuberculosis'])
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title('Confusion Matrix')
	plt.savefig('confusion_matrix.png')
	print("\nConfusion matrix plot saved to confusion_matrix.png")

	return ( svm_model, x_test, y_test )

def run():
	prepareModel()

if __name__ == "__main__":
	run()