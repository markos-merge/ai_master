from sklearn.svm import SVC
from basic_databases_manipulation import getSamplesFromLabels, produceShuffledMatrices
from basic_image_processing import to_hog
from open_file import load_CIFAR10, load_CIFAR10_testdata, images_to_numpy, load_label_names
import pickle
import numpy as np
import os
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm



def getClassifiersFromCSVFile( csv_file, probability = False ):
	# We need to gather the classifiers in order to retrain them, cause we saved the
	# the classifiers as a cuml.SVC class instead of sklearn.SVC 
	ret = {}
	with open( csv_file, "r" ) as f:
		lines = f.readlines()
		classifiers = {}
		flag = 0
		for line in lines:
			if flag == 0:
				flag += 1
				continue

			classifier_params = line.split( "," )
			ret[ classifier_params[0] ] = SVC( kernel = "rbf", C = float(classifier_params[1]), gamma = float(classifier_params[2]), probability=probability )

	return ret

def train_pair( i, j, x_train, y_train, clf ):
	label = str(i) + "vs" + str(j)

	clf.fit(x_train, y_train)

	return label, clf

def trainClassifiersFromCifar10( train_data_folder, classifiers, seed = 0 ):
	print("Loading and preparing CIFAR-10 data for training...")
	cifar10_data = load_CIFAR10( train_data_folder )
	x_data, y_data = images_to_numpy( cifar10_data, to_gray=True )
	x_data = to_hog( x_data )
	print("Data loaded and HOG features extracted.")

	tasks = []

	for name, classifier in classifiers.items():
		num = name.split( "vs" )
		i = int( num[0] )
		j = int( num[1] )
		x0, _ = getSamplesFromLabels(x_data, y_data, i)
		x1, _ = getSamplesFromLabels(x_data, y_data, j)
		
		x_train = np.vstack((x0, x1))
		y_train = np.hstack((np.ones(len(x0)) * -1, np.ones(len(x1))))
		x_train, y_train = produceShuffledMatrices( seed, x_train, y_train )

		
		task = delayed(train_pair)( i, j, x_train, y_train, classifier )
		tasks.append( task )

	results = Parallel(n_jobs=-1)(tqdm(tasks, desc="Training Classifiers"))

	for label, clf in results:
		classifiers[label] = clf

	return classifiers


def save_classifiers_to_folder( output_folder, classifiers ):
	os.makedirs(output_folder, exist_ok=True)
	for name, classifier in classifiers.items():
		model_filename = os.path.join(output_folder, f"svm_model_{name}.pkl")
		with open(model_filename, 'wb') as f:
			pickle.dump(classifier, f)
			print(f"Saved classifier {name} to {model_filename}")

def load_classifiers_from_folder( classifiers_folder ):
	classifiers = {}
	for filename in os.listdir(classifiers_folder):
		if filename.endswith(".pkl"):
			with open(os.path.join(classifiers_folder, filename), 'rb') as f:
				classifier = pickle.load(f)
				name = filename.replace("svm_model_", "").replace(".pkl", "")
				classifiers[name] = classifier
	return classifiers

def run_and_train_classifiers( probability = False ):
	train_data_folder = './cifar-10'
	output_folder = './results_sklearn_SVC'
	classifiers = getClassifiersFromCSVFile( "./results.csv", probability )
	trainClassifiersFromCifar10( train_data_folder, classifiers )
	save_classifiers_to_folder( output_folder, classifiers )

def test_classifiers(individual_results_filename = None):
	test_data_folder = './cifar-10'
	classifiers_folder = './results_sklearn_SVC'
	classifiers = load_classifiers_from_folder( classifiers_folder )
	cifar10_data = load_CIFAR10_testdata( test_data_folder )
	x_test, y_test = images_to_numpy( cifar10_data, to_gray=True )
	x_test = to_hog( x_test )

	for name, classifier in classifiers.items():
		num = name.split( "vs" )
		i = int( num[0] )
		j = int( num[1] )
		x0_test, _ = getSamplesFromLabels(x_test, y_test, i)
		x1_test, _ = getSamplesFromLabels(x_test, y_test, j)

		x_combined_test = np.vstack((x0_test, x1_test))
		y_combined_test = np.hstack((np.ones(len(x0_test)) * -1, np.ones(len(x1_test))))

		predictions = classifier.predict(x_combined_test)
		accuracy = np.mean(predictions == y_combined_test) * 100

		if individual_results_filename:
			with open( individual_results_filename, 'a' ) as f:
				f.write(f"Classifier {name} Test Accuracy: {accuracy:.2f}%\n")

def predictClassifier( i, j, classifier, image ):
	y = classifier.predict( image.reshape( 1, -1 ) )
	if y == 1:
		return j
	else:
		return i

def classify_image( classifiers, image, total_labels, cnt, probability = False, meta_classifier = None ):
	label_counting = np.zeros( total_labels )
	label_result = -1
	if meta_classifier and probability:
		label_result = classify_image_stacking(classifiers, meta_classifier, image)
	else:
		for name, classifier in classifiers.items():
			num = name.split( "vs" )
			i = int( num[0] )
			j = int( num[1] )
			if probability:
				y = classifier.predict_proba( image.reshape( 1, -1 ) )
				label_counting[i] += y[0][0]
				label_counting[j] += y[0][1]
			else:
				label_predicted = predictClassifier( i, j, classifier, image )
				label_counting[label_predicted] += 1
		label_result = label_counting.argmax()

	return ( cnt, label_result )

def predict_proba( classifiers, name, images ):
	return ( name, classifiers[name].predict_proba( images ) )

def train_meta_classifier(classifiers, x_data, y_data):
	print("Generating features for meta-classifier...")

	# This is a much faster, vectorized way to create the meta features.
	# Instead of iterating through each image, we predict probabilities for all images at once
	# for each classifier.

	# Ensure a consistent order for the features
	sorted_names = sorted(classifiers.keys())

	# Create a list of probability arrays, one for each classifier
	tasks = []
	for name in sorted_names:
		task = delayed(predict_proba)(classifiers, name, x_data)
		tasks.append( task )

	all_probs = Parallel(n_jobs=-1)(tqdm(tasks, desc="Getting Probabilities"))
	
	all_probs_dict = dict(all_probs)
	all_probs = [all_probs_dict[name] for name in sorted_names]
	# all_probs = [classifiers[name].predict_proba(x_data) for name in tqdm(sorted_names, desc="Getting Probabilities")]

	# Concatenate all probability arrays horizontally to form the meta-feature matrix
	meta_features = np.hstack(all_probs)

	X_meta = np.array(meta_features)

	print(f"Training meta-classifier on data of shape: {X_meta.shape}")
	meta_classifier = LogisticRegression(max_iter=1000, solver='liblinear')
	meta_classifier.fit(X_meta, y_data)
	print("Meta-classifier trained.")
	return meta_classifier

def classify_image_stacking(classifiers, meta_classifier, image):
	feature_vector = []
	for name in sorted(classifiers.keys()):
			probs = classifiers[name].predict_proba(image.reshape(1, -1))
			feature_vector.extend(probs[0])
	return meta_classifier.predict(np.array(feature_vector).reshape(1, -1))[0]

def get_str_if_found( found ):
	if found:
		return "PASSED"
	else:
		return "NOT PASSED"

def test_classifiers_whole( all_classifiers_filename = None , probabilty = False, use_meta_classifier = False ):
	test_data_folder = './cifar-10'
	classifiers_folder = './results_sklearn_SVC'

	cifar10_testdata = load_CIFAR10_testdata( test_data_folder )
	classifiers = load_classifiers_from_folder( classifiers_folder )
	meta_classifier = None
	if use_meta_classifier and probabilty:
		cifar10_data = load_CIFAR10( test_data_folder )
		x_train_meta, y_train_meta = images_to_numpy( cifar10_data, to_gray=True )
		x_train_meta = to_hog( x_train_meta )
		meta_classifier = train_meta_classifier(classifiers, x_train_meta, y_train_meta)
	x_test, y_test = images_to_numpy( cifar10_testdata, to_gray=True )
	x_test = to_hog( x_test )
	results_array = np.zeros( y_test.shape[0], dtype=np.int32 )
	tasks = []
	for i in range( x_test.shape[0] ):
		task = delayed(classify_image)( classifiers, x_test[i], 10, i, probabilty, meta_classifier )

		tasks.append( task )

	results = Parallel(n_jobs=-1)(tqdm(tasks, desc="Testing Classifiers"))

	for img_cnt, predicted_label in results:
		results_array[img_cnt] = predicted_label
	
	results = results_array.astype( np.uint32 )

	accuracy = np.mean( results == y_test ) * 100.

	if all_classifiers_filename:
		labels = load_label_names( test_data_folder )
		with open( all_classifiers_filename, 'a' ) as f:
			f.write(f"Total Accuracy: {accuracy:.2f}%\n")
			f.write(f"Detailed Results: \n")
			for i in range( x_test.shape[0] ):
				f.write( f"Predicted {labels[results[i]]}, Actual {labels[y_test[i]]}, {get_str_if_found( results[i] == y_test[i] )}\n" )


def run():
	# run_and_train_classifiers()
	# test_classifiers("inidividual_classifier_test_results.txt")
	# test_classifiers_whole("all_classifiers_test_results.txt")

	# run_and_train_classifiers( probability = True )
	# test_classifiers("individual_classifier_test_results.txt")
	test_classifiers_whole("all_classifiers_w_linear_regressor_test_results.txt", probabilty = True, use_meta_classifier= True )



if __name__ == "__main__":
	run()