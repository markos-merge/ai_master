import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp, options
from scipy.spatial.distance import cdist

class SVM:
	def __init__(self, kernel, gamma = 0.1, C = 1., pow = 1. ):
		self.C = C
		self.gamma = gamma
		if kernel == "linear":
			self.kernel = self._linear_kernel
		elif kernel == "poly":
			self.kernel = self._poly_kernel
			self.pow = pow
		elif kernel == "rbf":
			self.kernel = self._rbf_kernel
		else:
			raise ValueError( f"Unknown kernel type {kernel}")


	def _linear_kernel( self, x_0, x_1 ):
		return np.dot( x_0, x_1 )

	def _rbf_kernel( self, x_0, x_1 ):
		diff = x_0 - x_1
		return np.exp( -self.gamma*np.dot( diff, diff ) )
	
	def _poly_kernel( self, x_0, x_1 ):
		return ( np.dot( x_0, x_1 ) + 1 )**self.pow
	
	def fit( self, x, y ):
		n_samples, n_features = x.shape

		# --- OPTIMIZATION 1: Vectorized Gram Matrix Calculation ---
		# This replaces the slow nested Python loops with fast, compiled code.
		if self.kernel == self._linear_kernel:
			K = x @ x.T
		elif self.kernel == self._poly_kernel:
			K = (x @ x.T + 1) ** self.pow
		elif self.kernel == self._rbf_kernel:
			# Use scipy's cdist to compute squared euclidean distances efficiently
			sq_dists = cdist(x, x, 'sqeuclidean')
			K = np.exp(-self.gamma * sq_dists)

		P = matrix( np.outer( y, y )*K )
		q = matrix( np.ones( n_samples ) * -1 )

		A = matrix(y, (1, n_samples), 'd')
		b = matrix(0.0)

		G_upper = np.diag( np.ones( n_samples )*-1 )
		G_lower = np.identity( n_samples )
		G_np = np.vstack( ( G_upper, G_lower ) )
		G = matrix( G_np )
		h = np.zeros( 2*n_samples )
		h[0:n_samples] = 0
		h[n_samples:2*n_samples] = self.C

		G = matrix( G_np )
		h = matrix( h )

		options['show_progress'] = False
		solution = qp(P, q, G, h, A, b)
		alphas = np.ravel(solution['x'])
		sv_mask = alphas > 1e-5
		sv_indices = np.arange(len(alphas))[sv_mask]
		self.alphas = alphas[sv_mask]
		self.sv_x = x[sv_mask]
		self.sv_y = y[sv_mask]
		print(f"{len(self.alphas)} support vectors found.")

		# --- OPTIMIZATION 2: Vectorized Bias (b) Calculation ---
		# Calculate scores for all support vectors at once
		sv_K = K[sv_mask][:, sv_mask]
		scores = (self.alphas * self.sv_y) @ sv_K
		self.b = np.mean(self.sv_y - scores)

	def project(self, x_in):
		y_predict = np.zeros(len(x_in))
		for j in range( len( x_in) ):
			for i in range( len( self.alphas ) ):
				y_predict[j] += self.alphas[i]*self.sv_y[i]*self.kernel( self.sv_x[i], x_in[j] )

		return y_predict + self.b

	def predict(self, x_in):
		return np.sign(self.project(x_in))


if __name__ == "__main__":
# 1. Create a simple, linearly separable dataset
	np.random.seed(1)
	X = np.vstack([
		np.random.randn(20, 2) - [2, 2],
		np.random.randn(20, 2) + [2, 2]
	])
	y = np.hstack([np.ones(20) * -1, np.ones(20)])

	# 2. Create and train the SVM
	print("Training a linear SVM...")
	svm = SVM(kernel="rbf", C=1.0)
	svm.fit(X, y)

	# 3. Make predictions and check accuracy
	predictions = svm.predict(X)
	accuracy = np.mean(predictions == y) * 100
	print(f"Model Bias (b): {svm.b}")
	print(f"Training Accuracy: {accuracy:.2f}%")

