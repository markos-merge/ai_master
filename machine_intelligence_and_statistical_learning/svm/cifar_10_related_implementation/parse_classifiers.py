import pickle
# import numpy as np
# import sklearn as sk
import os
import re
from sklearn.svm import SVC

def getClassifiersCifar10( folder ):
	classifiers = {}

	for i in range( 10 ):
		for j in range( i + 1, 10 ):
			filename = "svm_model_" + str(i) + "vs" + str(j) + ".pkl"
			with open(folder + "/" + filename, 'rb') as f:
				classifiers[str(i) + "_" + str(j)] = pickle.load( f )
	
	return classifiers

def parse_report_file(file_path):
	"""
    Parses C, gamma, and score from a training report file.
    """
	try:
		with open(file_path, 'r') as f:
			content = f.read()

		# Use regular expressions to find C, gamma and score values
		params_match = re.search(r"Best Parameters found by GridSearchCV: \{'C': ([\d.]+), 'gamma': ([\d.]+)\}", content)
		score_match = re.search(r'GridSearchCV Best Score \(training set\): ([\d.]+)%', content)

		if params_match and score_match:
			C = float(params_match.group(1))
			gamma = float(params_match.group(2))
			score = float(score_match.group(1))
			return C, gamma, score
		else:
			return None, None, None
	except FileNotFoundError:
		print(f"Error: File not found at {file_path}")
		return None, None, None
	except Exception as e:
		print(f"Error: An error occurred: {e}")
		return None, None, None

def getHyperparametersForEachClassifier( folder ):
	hyperparameters = {}
	for filename in os.listdir(folder):
		if filename.startswith("svm_report_") and filename.endswith(".txt"):
			# Extract the name (e.g., "0vs1") from the filename
			name = filename.replace("svm_report_", "").replace(".txt", "")
			file_path = os.path.join(folder, filename)
			C, gamma, score = parse_report_file(file_path)
			if C is not None:
				hyperparameters[name] = {
					'C': C,
					'gamma': gamma,
					'score': score
				}
	return hyperparameters


def getSVMModels( folder ):
	models = {}

	hyperparams = getHyperparametersForEachClassifier( folder )

	for name, params in hyperparams.items():
		models[name] = SVC( kernel = "rbf", C = params['C'], gamma = params['gamma'] )

	return models


if __name__ == "__main__":
	hyperparams = getHyperparametersForEachClassifier("./results")
	print("name,C,gamma,score")
	for name, params in sorted(hyperparams.items()):
		print(f"{name},{params['C']:.2f},{params['gamma']:.2f},{params['score']:.2f}")
