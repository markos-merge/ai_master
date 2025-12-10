import pickle
import os
import sys

# Add the parent directory to the Python path to allow for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from open_file import load_CIFAR10_testdata, images_to_numpy
from basic_databases_manipulation import getSamplesFromLabels
from basic_image_processing import to_hog, save_image

def findFailedClassification( classifier_i, classifier_j, x_test, y_test, output_folder ):
	os.makedirs(output_folder, exist_ok=True)
	with open( os.path.join( results, f"svm_model_{classifier_i}vs{classifier_j}.pkl" ), 'rb' ) as f:
		classifier = pickle.load(f)
		x_class_i, _ = getSamplesFromLabels( x_test, y_test, classifier_i )
		x_class_j, _ = getSamplesFromLabels( x_test, y_test, classifier_j )

		x_class_i_transf = to_hog( x_class_i )
		x_class_j_transf = to_hog( x_class_j )

		for i in range( x_class_i_transf.shape[0] ):
			y_predict = classifier.predict( x_class_i_transf[i].reshape(1, -1) )
			if y_predict != -1:
				filename = f"wrong_classification_for_label_{classifier_i}_in_{classifier_i}vs{classifier_j}_{i}.png"
				save_image( x_class_i[i], os.path.join(output_folder, filename), is_grayscale=True )
				break
	
		for i in range( x_class_j_transf.shape[0] ):
			y_predict = classifier.predict( x_class_j_transf[i].reshape(1, -1) )
			if y_predict != 1:
				filename = f"wrong_classification_for_label_{classifier_j}_in_{classifier_i}vs{classifier_j}_{i}.png"
				save_image( x_class_j[i], os.path.join(output_folder, filename), is_grayscale=True )
				break

def findSuccessfulClassification( classifier_i, classifier_j, x_test, y_test, output_folder ):
	os.makedirs(output_folder, exist_ok=True)
	with open( os.path.join( results, f"svm_model_{classifier_i}vs{classifier_j}.pkl" ), 'rb' ) as f:
		classifier = pickle.load(f)
		x_class_i, _ = getSamplesFromLabels( x_test, y_test, classifier_i )
		x_class_j, _ = getSamplesFromLabels( x_test, y_test, classifier_j )

		x_class_i_transf = to_hog( x_class_i )
		x_class_j_transf = to_hog( x_class_j )

		for i in range( x_class_i_transf.shape[0] ):
			y_predict = classifier.predict( x_class_i_transf[i].reshape(1, -1) )
			if y_predict == -1:
				filename = f"successful_classification_for_label_{classifier_i}_in_{classifier_i}vs{classifier_j}_{i}.png"
				save_image( x_class_i[i], os.path.join(output_folder, filename), is_grayscale=True )
				break
	
		for i in range( x_class_j_transf.shape[0] ):
			y_predict = classifier.predict( x_class_j_transf[i].reshape(1, -1) )
			if y_predict == 1:
				filename = f"successful_classification_for_label_{classifier_j}_in_{classifier_i}vs{classifier_j}_{i}.png"
				save_image( x_class_j[i], os.path.join(output_folder, filename), is_grayscale=True )
				break

results = "./results_sklearn_SVC"
def main():
	test_data_folder = './cifar-10'
	cifar10_testdata = load_CIFAR10_testdata( test_data_folder )
	x_test, y_test = images_to_numpy( cifar10_testdata, to_gray=True )

	output_folder_failed = "failed_classifications"
	output_folder_successful = "successful_classifications"

	for i in range(10):
		for j in range(i + 1, 10):
			findFailedClassification(i, j, x_test, y_test, output_folder_failed)
			findSuccessfulClassification(i, j, x_test, y_test, output_folder_successful)

if __name__ == "__main__":
	main()