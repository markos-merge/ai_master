import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from cuml.decomposition import PCA
from pca_implementation import KernelPCA
from cuml.manifold import SpectralEmbedding
from cuml.manifold import UMAP
from cuml.cluster import SpectralClustering
from cuml.cluster import KMeans
from cuml.manifold import SpectralEmbedding
from cuml.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from open_file import load_CIFAR10, load_CIFAR10_testdata, images_to_numpy
import basic_image_processing as bip
from basic_image_processing import to_hog

def to_numpy( array ):
	if hasattr( array, 'get' ):
		return array.get()
	
	return array

def visualize_clusters(embedding, labels):
	embedding = to_numpy( embedding )
	labels = to_numpy( labels )

	plt.figure(figsize=(10, 8))
	scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='tab10', s=1, alpha=0.5)
	plt.colorbar(scatter, label='Cluster Label')
	plt.title('Spectral Clustering Visualization')
	plt.savefig( 'spectral_clustering_visualization.png' )
	plt.show()

def run():
	images = load_CIFAR10( './cifar-10' )
	test_images = load_CIFAR10_testdata( './cifar-10' )

	x, y = images_to_numpy( images, to_gray = False )

	x_hog = to_hog( x, pixels_per_cell=(4, 4), cells_per_block=(2, 2), is_grayscale = False, include_intensity_image = True )

	kpca = KernelPCA( n_components = 10, kernel = 'rbf', gamma = None )
	x_kpca = kpca.fit_transform( x_hog )

	model = SpectralClustering( n_clusters = 10, random_state=42, n_neighbors = 50 )

	labels = model.fit_predict( x_kpca )

	embedding_model = SpectralEmbedding( n_neighbors = 50, random_state = 42 )
	embedding = embedding_model.fit_transform( x_kpca )

	visualize_clusters( embedding, labels )

	labels_numpy = to_numpy( labels )

	ari = adjusted_rand_score( y, labels_numpy )
	nmi = normalized_mutual_info_score( y, labels_numpy )
	print( f"Adjusted Rand Index (ARI): {ari:.4f}" )
	print( f"Normalized Mutual Information (NMI): {nmi:.4f}" )

if __name__ == "__main__":
	run()