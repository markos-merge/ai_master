import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
	name = "cleaned_data.csv"
	df = pd.read_csv( name )
	
	df_hh_gt = pd.read_csv( "data/train_hh_gt.csv" )
	merged_df = df.merge( df_hh_gt[['hhid', 'cons_ppp17']], on='hhid' )
	
	# Calculate correlations
	correlations = merged_df.drop(columns=['hhid', 'survey_id', 'weight'], errors='ignore').corr()['cons_ppp17'].sort_values(ascending=False)
	correlations = correlations.drop('cons_ppp17')

	print(correlations)

	plt.figure(figsize=(12, 8))
	sns.barplot(x=correlations.index, y=correlations.values)
	plt.xticks(rotation=90)
	plt.title("Feature Correlations with cons_ppp17")
	plt.tight_layout()
	plt.savefig("correlations_barplot.png")
	plt.show()

	threshold = 0.15
	selected_features = correlations[abs(correlations) >= threshold].index.tolist()
	meta_cols = [col for col in ['hhid', 'survey_id', 'weight'] if col in merged_df.columns]
	columns_to_keep = meta_cols + selected_features
	filtered_df = merged_df[columns_to_keep]
	filtered_df.to_csv("cleaned_data_filtered_from_correlation.csv", index=False)
	print(f"Saved filtered data to cleaned_data_filtered.csv. Kept {len(selected_features)} features.")