import numpy as np
from scipy import linalg as la

def zeromean_gaussian(X,ls):
    """Compute the unnormalised gaussian values at locations specified in X, for a Gaussian
    centred at the origin with covariance a diagonal with values ls^2."""
    twotimesls2 = 2*ls**2
    return np.exp(-np.sum((X**2),1)/twotimesls2)

def findbound(X,W,ls,d,gridspacing,gridsize,ignorenegatives=False):
    """
    The centres of the n gaussians in the mixture model are defined as locations in the nxd X matrix
    W is a vector of weights (nx1).
    ls = scalar lengthscale
    d = number of dimensions (usually X.shape[1])
    gridspacing = how far apart the grid squares should be
    gridsize = scalar - maximum value of grid in all dimensions. Grid starts at origin.
    ignorenegatives = set to true to have the negative weights set to zero.
    This is necessary if you've performed dimensionality reduction as the negative values may have moved closer to the original datapoints, thus reducing the computed maxima.
    
    The gaussians here aren't normalised. please take this into account when chosing W.
    """
    mg = np.meshgrid(*([np.arange(0,gridsize,gridspacing)]*d))
    mesh = []
    for mgitem in mg:
        mesh.append(mgitem.flatten())
    mesh = np.array(mesh).T
    tot = np.zeros(len(mesh))
    
    #disable the weights that are negative.
    newW = W.copy()
    if ignorenegatives:
        newW[newW<0] = 0
    for i,(x,w) in enumerate(zip(X,newW)):
        tot += w*zeromean_gaussian(mesh-x,ls)
    maxgridpoint = np.max(tot)
    #compute possible additional height between grid points
    p = np.sqrt(d)*gridspacing/2 
    potential_shortfall = (1-zeromean_gaussian(np.array([[p]]),ls))*np.sum(np.abs(W))
    return maxgridpoint+potential_shortfall



    
def PCA(data, dims_rescaled_data=2):
    """
    Based on https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python

    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    means = data.mean(axis=0)
    data -= means
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix, 
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return (evecs.T @ data.T).T, evals, evecs, means
