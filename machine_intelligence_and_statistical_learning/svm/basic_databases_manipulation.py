import numpy as np

def getSamplesFromLabels( X, Y, label ):
	mask_label = Y == label
	
	return X[mask_label], Y[mask_label]



def produceShuffledMatrices(seed, x, y):
	np.random.seed(seed)
	shuffled_indices = np.random.permutation( len(x) )

	return x[shuffled_indices], y[shuffled_indices]
