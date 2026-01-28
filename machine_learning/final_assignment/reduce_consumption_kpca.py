import pandas as pd
import numpy as np
import sys
import os

if __name__ == "__main__":
	print("Loading cleaned data...")
	try:
			df = pd.read_csv("cleaned_data.csv")
	except FileNotFoundError:
			print("Error: 'cleaned_data.csv' not found. Please run 'clean_data.py' first.")
			exit(1)

	print("Skipping Kernel PCA as requested. Using original features.")
	
	# Simply copy the dataframe to the output variable
	# This ensures downstream scripts receive the full feature set
	df_compressed = df.copy()

	output_filename = "cleaned_data_compressed.csv"
	df_compressed.to_csv(output_filename, index=False)
	
	print(f"Saved compressed dataset to '{output_filename}'.")
	print(f"Original columns: {df.shape[1]}, New columns: {df_compressed.shape[1]}")
	print("You can now run 'svm_training.py'.")

	# Process Test Data if it exists
	test_filename = "cleaned_data_test.csv"
	if os.path.exists(test_filename):
		print(f"Processing {test_filename}...")
		df_test = pd.read_csv(test_filename)
		df_test_compressed = df_test.copy()
		df_test_compressed.to_csv("cleaned_data_test_compressed.csv", index=False)
		print("Saved compressed test dataset.")
