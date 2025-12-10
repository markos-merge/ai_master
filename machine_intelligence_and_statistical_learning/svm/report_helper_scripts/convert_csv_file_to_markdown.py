import pandas as pd

def convert_csv_to_markdown_table(csv_file_path, output_file_path=None):
	try:
			df = pd.read_csv(csv_file_path)

			markdown_table = "|" + "|".join(df.columns) + "|\n"
			markdown_table += "|" + "|".join(["---"] * len(df.columns)) + "|\n"

			for index, row in df.iterrows():
					markdown_table += "|" + "|".join(str(item) for item in row) + "|\n"

			if output_file_path:
					with open(output_file_path, 'w') as f:
							f.write(markdown_table)
					print(f"Markdown table saved to: {output_file_path}")
			else:
					print(markdown_table)

	except FileNotFoundError:
			print(f"Error: CSV file not found at '{csv_file_path}'")
	except Exception as e:
			print(f"An error occurred: {e}")


def convert_cifar_10_labels_in_csv_file( csv_file_path ):
	label_names = {
		0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
		5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
	}
	try:
		df = pd.read_csv(csv_file_path)

		df['Readable_Labels'] = df['name'].apply(
			lambda x: f"{label_names[int(x.split('vs')[0])]} vs {label_names[int(x.split('vs')[1])]}"
		)

		cols = df.columns.tolist()
		cols.insert(1, cols.pop(cols.index('Readable_Labels')))
		df = df[cols]

		output_csv_path = csv_file_path.replace(".csv", "_with_labels.csv")
		df.to_csv(output_csv_path, index=False)
		print(f"CSV with readable labels saved to: {output_csv_path}")
		return output_csv_path

	except FileNotFoundError:
		print(f"Error: CSV file not found at '{csv_file_path}'")
		return None
	except Exception as e:
		print(f"An error occurred: {e}")
		return None
		

if __name__ == "__main__":
	csv_input_file = convert_cifar_10_labels_in_csv_file( './results.csv' )
	markdown_output_file = 'results_table.md'

	print("--- Markdown Table (Console Output) ---")
	convert_csv_to_markdown_table(csv_input_file)

	print("\n--- Markdown Table (File Output) ---")
	convert_csv_to_markdown_table(csv_input_file, markdown_output_file)