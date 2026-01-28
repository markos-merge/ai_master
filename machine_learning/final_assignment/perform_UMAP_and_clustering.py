from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pca_implementation import KernelPCA
import os

def perform_umap_and_hdbscan(df, n_neighbors=15, min_dist=0.1, n_components=2, min_cluster_size=100 ):
	# Separate hhid before UMAP and clustering
	hhid = df['hhid']
	gt = df['cons_ppp17']
	survey_id = df['survey_id'] if 'survey_id' in df.columns else None
	features_df = df.drop(columns=['hhid', 'cons_ppp17', 'survey_id', 'weight'], errors='ignore')

	# kpca = KernelPCA(n_components = kpca_components, kernel='rbf', gamma=0.05, max_samples = 1000 )
	# features_df = features_df.to_numpy()
	# features_df = kpca.fit_transform(features_df)
	# features_df = features_df.get()

	reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components = n_components, n_jobs = -1 )
	umap_embedding = reducer.fit_transform(features_df)

	# Add UMAP embeddings to the DataFrame
	for i in range(n_components):
		df[f'UMAP_{i+1}'] = umap_embedding[:, i]

	# Perform HDBSCAN clustering
	clusterer = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
	df['cluster'] = clusterer.fit_predict(umap_embedding)

	# Add hhid back
	df['hhid'] = hhid
	df['cons_ppp17'] = gt
	if survey_id is not None:
		df['survey_id'] = survey_id
	
	return df, reducer, clusterer

def saveClustersToFile(folder, df):
	cluster_values = np.unique( df['cluster'].to_numpy() )

	for i in cluster_values:
		if i == -1:
			continue

		df_cluster = df[ df['cluster'] == i ]
		df_cluster.to_csv( f"{folder}/cluster_{i}.csv", index = False )



def plot_umap_clusters(df, title="UMAP Clustering"):
	if 'UMAP_1' not in df.columns or 'UMAP_2' not in df.columns or 'cluster' not in df.columns:
			raise ValueError("DataFrame must contain 'UMAP_1', 'UMAP_2', and 'cluster' columns for plotting.")

	plt.figure(figsize=(12, 10))
	sns.scatterplot(
			x='UMAP_1',
			y='UMAP_2',
			hue='cluster',
			data=df,
			palette='viridis',
			s=50,
			alpha=0.7,
			legend='full'
	)

	if 'cons_ppp17' in df.columns:
		stats = df.groupby('cluster')['cons_ppp17'].agg(['mean', 'std'])
		centroids = df.groupby('cluster')[['UMAP_1', 'UMAP_2']].mean()

		for cluster_id in stats.index:
			if cluster_id == -1:
				continue
			
			mean_val = stats.loc[cluster_id, 'mean']
			std_val = stats.loc[cluster_id, 'std']
			x, y = centroids.loc[cluster_id]
			plt.text(x, y, f"{mean_val:.2f}\n±{std_val:.2f}", horizontalalignment='center', verticalalignment='center', size=9, weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

	plt.title(title, fontsize=16)
	plt.xlabel('UMAP_1', fontsize=12)
	plt.ylabel('UMAP_2', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.show()

if __name__ == "__main__":
	# Example Usage:
	# Create a dummy DataFrame for demonstration
	dummy_df = pd.read_csv( "cleaned_data_filtered_from_correlation.csv" )

	# Load ground truth labels
	df_hh_gt = pd.read_csv( "data/train_hh_gt.csv" )
	dummy_df = dummy_df.merge( df_hh_gt[['hhid', 'cons_ppp17']], on='hhid' )

	print("Original DataFrame head:")
	print(dummy_df.head())

	# Perform UMAP and HDBSCAN
	clustered_df, umap_model, hdbscan_model = perform_umap_and_hdbscan(dummy_df.copy(), n_components=2)

	print("\nClustered DataFrame head:")
	print(clustered_df.head())
	print(f"\nNumber of clusters found: {clustered_df['cluster'].nunique()}")
	print(f"Cluster distribution:\n{clustered_df['cluster'].value_counts()}")
	
	# Calculate and print mean and std for each cluster
	cluster_stats = clustered_df.groupby('cluster')['cons_ppp17'].agg(['mean', 'std', 'count'])
	print("\nCluster Statistics (cons_ppp17):")
	print(cluster_stats)

	os.makedirs( "clusters/", exist_ok=True )
	saveClustersToFile( "clusters/", clustered_df )



	# Plot the UMAP clusters
	plot_umap_clusters(clustered_df, title="UMAP Clustering of Dummy Data")
