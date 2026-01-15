import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.datasets import fetch_openml
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pca_implementation import KernelPCA
import matplotlib.pyplot as plt

def to_numpy( array ):
	if hasattr( array, 'get' ):
		return array.get()
	
	return array

def visualize_clusters(embedding, labels, true_labels, title='Spectral Clustering Visualization', ax=None):
	embedding = to_numpy( embedding )
	labels = to_numpy( labels )

	if ax is None:
		fig, ax = plt.subplots(figsize=(10, 8))

	scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=1, alpha=0.5)
	ax.set_title(title)

	unique_labels = np.unique( true_labels )
	for label in unique_labels:
		if label == -1: continue
		centroid = embedding[true_labels == label].mean(axis=0)
		ax.text(centroid[0], centroid[1], str(label), fontsize=10, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

	if ax is None:
		plt.show()

def train():
	data_file = 'mnist_related_implementation/mnist.npz'
	if os.path.exists(data_file):
		print(f"Loading MNIST from {data_file}...")
		with np.load(data_file) as data:
			X = data['X']
			y = data['y']
	else:
		print("Loading MNIST data...")
		mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
		X = mnist.data.reshape(-1, 28, 28)
		y = mnist.target.astype(int)
		np.savez(data_file, X=X, y=y)
		print(f"Saved MNIST data to {data_file}")

	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	ari_scores = []
	nmi_scores = []

	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	axes = axes.flatten()


	for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
		print(f"Processing Fold {fold + 1}...")
		
		x_fold = X[train_idx].reshape(len(train_idx), -1) / 255.0
		y_fold = y[train_idx]

		pipeline = Pipeline([
			( 'kpca', KernelPCA( n_components = 150, kernel='rbf', gamma=0.001, max_samples = 6000 ) ),
			( 'umap', UMAP( n_components=2, random_state=42, n_neighbors=15, min_dist=0.0, init='spectral' ) ),
			( 'hdbscan', HDBSCAN( min_samples=10, min_cluster_size = 100, cluster_selection_method='eom' ) )
		] )

		labels = pipeline.fit_predict(x_fold)
		labels_np = to_numpy(labels)

		ari = adjusted_rand_score(y_fold, labels_np)
		nmi = normalized_mutual_info_score(y_fold, labels_np)
		ari_scores.append(ari)
		nmi_scores.append(nmi)
		print(f"Fold {fold + 1} - ARI: {ari:.4f}, NMI: {nmi:.4f}")

		x_umap = pipeline[:-1].transform(x_fold)
		visualize_clusters(x_umap, labels_np, y_fold, title=f'Fold {fold+1} (ARI: {ari:.2f})', ax=axes[fold])

	if len(axes) > 5:
		fig.delaxes(axes[5])

	print(f"\nAverage ARI: {np.mean(ari_scores):.4f} +/- {np.std(ari_scores):.4f}")
	print(f"Average NMI: {np.mean(nmi_scores):.4f} +/- {np.std(nmi_scores):.4f}")

	plt.tight_layout()
	plt.savefig('UMAP_CV_Subplots.png')
	plt.show()

if __name__ == '__main__':
	train()