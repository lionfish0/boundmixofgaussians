import numpy as np
from scipy import linalg as la

def zeromean_gaussian_1d(x,ls):
    """Compute the unnormalised gaussian values at locations specified in X, for a Gaussian
    centred at the origin with variance ls^2"""
    twotimesls2 = 2*ls**2
    return np.exp(-(x**2)/twotimesls2)
    
def zeromean_gaussian(X,ls):
    """Compute the unnormalised gaussian values at locations specified in X, for a Gaussian
    centred at the origin with covariance a diagonal with values ls^2."""
    twotimesls2 = 2*ls**2
    return np.exp(-np.sum((X**2),1)/twotimesls2)

def findbound_lowdim(X,W,ls,d,gridspacing,gridstart,gridend,ignorenegatives=False):
    """
    The centres of the n gaussians in the mixture model are defined as locations in the nxd X matrix
    W is a vector of weights (nx1).
    ls = scalar lengthscale
    d = number of dimensions (usually X.shape[1])
    gridspacing = how far apart the grid squares should be
    gridstart/end = list of start and end values
    ignorenegatives = set to true to have the negative weights set to zero.
    This is necessary if you've performed dimensionality reduction as the negative values may have moved closer to the original datapoints, thus reducing the computed maxima.
    
    The gaussians here aren't normalised. please take this into account when chosing W.
    """
    #print(X,W,ls,d,gridspacing,gridstart,gridend,ignorenegatives)
    
    assert len(gridstart)==d, "Gridstart & gridend should have same number of items as the number of dimensions (%d)" % d
    meshlist = []
    for start,end in zip(gridstart,gridend):
        meshlist.append(np.arange(start,end,gridspacing))
    mg = np.meshgrid(*meshlist) #note: numba doesn't like the * thing
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


def findbound(X,W,ls,d,gridspacing,gridstart,gridend,fulldim=False,forceignorenegatives=False,dimthreshold=3):
    assert len(gridstart)==d, "Gridstart & gridend should have same number of items as the number of dimensions (%d)" % d
    if X.shape[1]>dimthreshold and not fulldim:
        #print("Compacting to 3d manifold...")
        lowd = dimthreshold
        lowdX,evals,evecs,means = PCA(X.copy(),lowd)
        ignorenegatives = True
        
        gridwidths = (gridend-gridstart)/2
        ##TODO Print warning if not equal as it might be we're handling something that's very "non-spherical"
        radius = np.sqrt(np.sum(gridwidths**2)) #largest radius
        centre = gridstart+gridwidths/2
        lowdcentre = (evecs.T @ centre.T).T
        gridstart = lowdcentre-radius
        gridend = lowdcentre+radius
    else:
        lowdX = X
        lowd = X.shape[1]
        ignorenegatives = forceignorenegatives
        
    #TODO should we subtract/add gridspacing to the gridstart/gridend
    return findbound_lowdim(lowdX,W,ls=ls,d=lowd,gridspacing=gridspacing,gridstart=gridstart,gridend=gridend,ignorenegatives=ignorenegatives)


    
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
