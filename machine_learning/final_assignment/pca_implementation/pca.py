import cupy as cp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import numpy as np

class PCA:
	def __init__( self, n_components ):
		self.n_components = n_components
		self.mean = None
		self.components = None

	def fit( self, X ):
		X = cp.asarray(X)
		self.mean = cp.mean( X, axis=0 )
		X_centered = X - self.mean

		covariance_matrix = cp.cov( X_centered, rowvar=False )

		eigenvalues, eigenvectors = cp.linalg.eigh( covariance_matrix )

		sorted_indices = cp.argsort( eigenvalues )[::-1]
		sorted_eigenvalues = eigenvalues[sorted_indices]
		sorted_eigenvectors = eigenvectors[:, sorted_indices]

		if self.n_components < 1.:
			total_variance = cp.sum( sorted_eigenvalues )
			variance_threshold = self.n_components * total_variance
			cumulative_variance = cp.cumsum( sorted_eigenvalues )
			self.n_components = cp.searchsorted( cumulative_variance, variance_threshold ) + 1

			if self.n_components > sorted_eigenvectors.shape[1]:
				self.n_components = sorted_eigenvectors.shape[1]

		self.components = sorted_eigenvectors[:, :self.n_components]

	def transform(self, X):
		X = cp.asarray(X)
		X_centered = X - self.mean

		return cp.dot(X_centered, self.components)

	def fit_transform( self, X ):
			self.fit(X)
			return self.transform(X)
	

class KernelPCA( BaseEstimator, TransformerMixin ):
	def __init__( self, n_components = 100, kernel='rbf', gamma=None, max_samples=12000 ):
		self.n_components = n_components
		self.kernel = kernel
		self.gamma = gamma
		self.max_samples = max_samples
	


	def _compute_kernel( self, X ):
		if self.kernel == 'linear':
			return cp.dot(X, X.T)
		elif self.kernel == 'rbf':
			X_norm = cp.sum(X ** 2, axis=1).reshape(-1, 1)
			K = cp.exp(-self.gamma_ * (X_norm + X_norm.T - 2 * cp.dot(X, X.T)))
			
			return K
		else:
			raise ValueError(f"Unsupported kernel: {self.kernel}")

	def _compute_centered_kernel_matrix( self, X ):
		centered_mat = self.K_
	
		num_samples = X.shape[0]
		mean_cols = cp.sum(self.K_, axis=0) / num_samples
		mean_rows = (cp.sum(self.K_, axis=1) / num_samples).reshape(-1, 1)
		grand_mean = cp.sum(self.K_) / (num_samples * num_samples)
		centered_mat = self.K_ - mean_cols - mean_rows + grand_mean

		return centered_mat

	def fit( self, X, y = None):
		n_samples = X.shape[0]
		if n_samples > self.max_samples:
			# Select random indices
			rng = np.random.RandomState(42)
			indices = rng.choice(n_samples, self.max_samples, replace=False)
			X_fit = X[indices]
		else:
			X_fit = X
		# X = cp.asarray(X)
		self.X_fit_ = cp.asarray(X_fit, dtype=cp.float32)
		self.n_features_in_ = self.X_fit_.shape[1]

		if self.gamma is None:
			self.gamma_ = 1.0 / self.X_fit_.shape[1]
		else:
			self.gamma_ = self.gamma

		self.K_ = self._compute_kernel( self.X_fit_ )
		self.train_row_means_ = cp.mean(self.K_, axis=1)
		self.centered_mat_ = self._compute_centered_kernel_matrix( self.X_fit_ )
		eigenvalues, eigenvectors = cp.linalg.eigh( self.centered_mat_ )
		sorted_indices = cp.argsort( eigenvalues )[::-1]
		sorted_eigenvalues = eigenvalues[sorted_indices]
		sorted_eigenvectors = eigenvectors[:, sorted_indices]
		self.alphas_ = sorted_eigenvectors[:, :self.n_components]
		self.lambdas_ = sorted_eigenvalues[:self.n_components]
		self.alphas_ = self.alphas_ / cp.sqrt(self.lambdas_ + 1e-10)


		return self

	def _compute_kernel_between( self, X1, X2 ):
		if self.kernel == 'linear':
			return cp.dot(X1, X2.T)
		elif self.kernel == 'rbf':
			X1_norm = cp.sum(X1 ** 2, axis=1).reshape(-1, 1)
			X2_norm = cp.sum(X2 ** 2, axis=1).reshape(1, -1)
			K = cp.exp(-self.gamma_ * (X1_norm + X2_norm - 2 * cp.dot(X1, X2.T)))
			return K
		else:
			raise ValueError(f"Unsupported kernel: {self.kernel}")

	def _compute_center_kernel_vector( self, X ):
		K_new = self._compute_kernel_between( X, self.X_fit_ )
		centered_K_vector = K_new - self.train_row_means_

		return centered_K_vector

	def transform( self, X ):
		is_gpu = hasattr(X, 'device') or 'cupy' in str(type(X))
        
		if not is_gpu:
			X = check_array(X, accept_sparse=False, ensure_2d=True)
			X = cp.asarray(X)
		else:
			check_is_fitted(self, ['n_features_in_'])
			if X.shape[1] != self.n_features_in_:
				raise ValueError(f"X has {X.shape[1]} features, but KernelPCA is expecting {self.n_features_in_} features.")
			X = cp.asarray(X)
	
		check_is_fitted( self, ['X_fit_', 'alphas_', 'train_row_means_'] )
		X = cp.asarray( X )
		centered_K_vector = self._compute_center_kernel_vector( X )

		return cp.dot( centered_K_vector, self.alphas_ )

import unittest

class TestKernelPCA(unittest.TestCase):

    def setUp(self):
        # Create simple 2D synthetic data
        # Two distinct clusters for easy separation
        self.X_train = cp.array([
            [1.0, 2.0], [1.1, 2.1],  # Cluster 1
            [10.0, 20.0], [10.1, 20.1] # Cluster 2
        ])
        self.n_samples = self.X_train.shape[0]

    def test_linear_kernel_computation(self):
        """Test if linear kernel computes standard dot product."""
        kpca = KernelPCA(n_components=1, kernel='linear')
        K_computed = kpca._compute_kernel(self.X_train)
        
        # Manual calculation: K = XX^T
        K_expected = cp.dot(self.X_train, self.X_train.T)
        
        cp.testing.assert_array_almost_equal(K_computed, K_expected)

    def test_rbf_kernel_mathematics(self):
        """Test RBF kernel values against manual calculation."""
        # Simple data: 2 points on a line
        X = cp.array([[0.0], [1.0]]) 
        gamma = 0.5
        kpca = KernelPCA(n_components=1, kernel='rbf', gamma=gamma)
        
        K = kpca._compute_kernel(X)
        
        # Expected:
        # Diagonal (dist=0) -> exp(0) = 1.0
        # Off-diagonal (dist^2=1) -> exp(-0.5 * 1) = 0.6065...
        expected_val = cp.exp(-gamma * 1.0)
        
        self.assertAlmostEqual(K[0, 0], 1.0)
        self.assertAlmostEqual(K[0, 1], expected_val)

    def test_centering_logic_pdf_compliant(self):
        """
        Verify the centering formula matches the PDF:
        Psi^T Psi = K - 1_M K - K 1_M + 1_M K 1_M
        """
        kpca = KernelPCA(n_components=2, kernel='linear')
        
        # Create a dummy kernel matrix
        K = cp.array([[10., 20.], [30., 40.]])
        kpca.K_ = K # Mock the internal K
        
        # Run the centering method
        K_centered = kpca._compute_centered_kernel_matrix(K)
        
        # Manual Calculation
        # Row Means: [15, 35]
        # Col Means: [20, 30]
        # Grand Mean: 25
        # Element (0,0): 10 - 20 (col) - 15 (row) + 25 (grand) = 0
        
        expected_00 = 10 - 20 - 15 + 25
        self.assertAlmostEqual(K_centered[0, 0], expected_00)

    def test_fit_attributes_shapes(self):
        """Check if fit populates attributes with correct shapes."""
        n_components = 2
        kpca = KernelPCA(n_components=n_components, kernel='linear')
        kpca.fit(self.X_train)
        
        # Check Alphas (Eigenvectors)
        self.assertEqual(kpca.alphas_.shape, (self.n_samples, n_components))
        
        # Check Lambdas (Eigenvalues)
        self.assertEqual(kpca.lambdas_.shape, (n_components,))
        
        # Check Row Means (Stored for projection)
        self.assertEqual(kpca.train_row_means_.shape, (self.n_samples,))

    def test_projection_logic_pdf_compliant(self):
        """
        Verify the transform step strictly follows the PDF:
        vector = k(x, u_i) - k(u_i, mean)
        """
        kpca = KernelPCA(n_components=1, kernel='linear')
        kpca.fit(self.X_train)
        
        # Create a test point identical to the first training point
        X_test = self.X_train[0:1] # Shape (1, 2)
        
        # Manually compute the centered vector
        # 1. Raw Kernel between Test and Train
        k_raw = cp.dot(X_test, self.X_train.T)
        
        # 2. Subtract stored row means (C value)
        expected_vector = k_raw - kpca.train_row_means_
        
        # 3. Use the class method
        computed_vector = kpca._compute_center_kernel_vector(X_test)
        
        cp.testing.assert_array_almost_equal(computed_vector, expected_vector)

    def test_end_to_end_linear_matches_pca(self):
        """
        Sanity Check: Linear Kernel PCA should produce results 
        equivalent to Standard PCA (up to sign flipping).
        """
        # 1. Standard PCA (via Covariance Matrix manually)
        X_centered = self.X_train - self.X_train.mean(axis=0)
        cov = cp.dot(X_centered.T, X_centered) / (self.n_samples - 1)
        eigvals, eigvecs = cp.linalg.eigh(cov)
        
        # Sort and project standard PCA
        idx = cp.argsort(eigvals)[::-1]
        standard_pca_proj = cp.dot(X_centered, eigvecs[:, idx][:, :1])
        
        # 2. Your Kernel PCA
        kpca = KernelPCA(n_components=1, kernel='linear')
        kpca_proj = kpca.fit_transform(self.X_train)
        
        # Compare magnitude (absolute values) because signs might flip 
        # between implementations (eigenvectors direction is arbitrary)
        # Note: Kernel PCA output scales with sqrt(eigenvalue) * N factor diffs 
        # usually exist depending on normalization. 
        # We just check correlation here to ensure structural correctness.
        
        correlation = cp.corrcoef(standard_pca_proj.flatten(), kpca_proj.flatten())[0, 1]
        self.assertTrue(abs(correlation) > 0.99, "Linear KPCA should perfectly correlate with PCA")

from sklearn.utils.estimator_checks import check_estimator
if __name__ == '__main__':
	check_estimator( KernelPCA() )
	unittest.main(argv=['first-arg-is-ignored'], exit=False)


		