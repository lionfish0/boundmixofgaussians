import numpy as np
from scipy import linalg as la

def zeromean_gaussian_1d(x,ls,v):
    """Compute the unnormalised gaussian values at locations specified in X, for a Gaussian
    centred at the origin with height v, lengthscale ls"""
    twotimesls2 = 2*ls**2
    return v*np.exp(-(x**2)/twotimesls2)
    
def zeromean_gaussian(X,ls,v):
    """Compute the unnormalised gaussian values at locations specified in X, for a Gaussian
    centred at the origin with covariance a diagonal with values ls^2."""
    twotimesls2 = 2*ls**2
    return v*np.exp(-np.sum((X**2),1)/twotimesls2)

def findbound_lowdim(X,W,ls,v,d,gridspacing,gridstart,gridend,ignorenegatives=False):
    """
    The centres of the n gaussians in the mixture model are defined as locations in the nxd X matrix
    W is a vector of weights (nx1).
    ls = scalar lengthscale
    v = kernel variance
    d = number of dimensions (usually X.shape[1])
    gridspacing = how far apart the grid squares should be
    gridstart/end = list of start and end values
    ignorenegatives = set to true to have the negative weights set to zero.
    This is necessary if you've performed dimensionality reduction as the negative values may have moved closer to the original datapoints, thus reducing the computed maxima.
    
    The gaussians here aren't normalised. please take this into account when choosing W.
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
        tot += w*zeromean_gaussian(mesh-x,ls,v)
    maxgridpoint = np.max(tot)
    #compute possible additional height between grid points
    p = np.sqrt(d)*gridspacing/2 
    #potential_shortfall = (v-zeromean_gaussian(np.array([[p]]),ls,v))*np.sum(np.abs(W))
    #return maxgridpoint+potential_shortfall
    
    maxgridpoint/= zeromean_gaussian_1d(np.array([p]),ls,1)
    #print(gridstart,gridend,X.T,W,maxgridpoint)
    return maxgridpoint
    
    



### New code to make use of negative weights.
np.set_printoptions(suppress=True,precision=5)
from boundmixofgaussians import zeromean_gaussian_1d

def compute_sum(w1,w2,x,ls,dist):
    return zeromean_gaussian_1d(x,ls,w1)+zeromean_gaussian_1d(x-dist,ls,w2)
def compute_grad(w1,w2,x,ls,dist):
    return (((x/ls**2)*zeromean_gaussian_1d(x,ls,w1))+(((x-dist)/ls**2)*zeromean_gaussian_1d(x-dist,ls,w2)))


def findpeak(w1,w2,dist,ls):
    """
    Find the peak location and value of the sum of a positive and a negative
    weighted pair of Gaussians, with the positive Gaussian centred at x=0, and
    the negatively weighted Gaussian centred at x=+dist.
       w1 is positive weighted, w2 is negative weighted. 
       ls is the lengthscale.
       
    Returns the location of the peak of the sum, and its value.
    
 resulting    |
 location of  |.............
 sum's peak ..|................w1=0.5
     .........|.......................        dist=2
    ----------|------+------------------------+------------------------------
           -0.42  ....................................................
                     ..........................................
      ls=1.0             ...................................
                             ...........................w2=-1.5
                                 ..................
      
    For example:
        x, p = findpeak(0.5,-1.5,2.0,1.0)
    will have x=-0.42, p=0.377
    """
    #print(w1,w2,dist,ls)
    assert w1>0
    assert w2<0
    x = 0
    lr = 0.7
    #for it in range(20):
    oldx = np.inf
    while np.abs(oldx-x)>1e-3: #TODO Pick something sensible for this
        oldx = x
        x-=lr*compute_grad(w1,w2,x,ls,dist)
        lr*=0.97 #TODO and this
        
    return x, compute_sum(w1,w2,x,ls,dist)

def mergenegatives(X,W,ls):
    
    i = -1
    #for i,(x,w) in enumerate(zip(X,W)): #find all negatives
    #print("|",end="")
    while(i<len(X)-1):

        i+=1
        x = X[i,:]
        w = W[i]   
        
        if w>=0:
            continue
        #print(",",end="")            
        sqrdists = np.sum((x-X)**2,1)
        sqrdists[W<=0] = np.infty #we want to combine with a positive weight
        nearestpositive = np.argmin(sqrdists)
        dist = np.sqrt(sqrdists[nearestpositive])
        if dist==np.infty: #all points are now negative!
            #print(">>>",end="")
            break
        offset,newpeak = findpeak(W[nearestpositive],w,dist,ls)
        #print(offset,newpeak)
        vector = (x - X[nearestpositive,:])
        #offset value is positive in direction from +ve weight to -ve weight
        newx = (X[nearestpositive,:] + offset * vector/dist)
        #delete nearestpositive and i from X and W
        #add newx and newpeak to X and W
        #do so by replacing one, deleting other
        #
        #so in effect, move +ve one:
        X[nearestpositive,:] = newx
        W[nearestpositive] = newpeak
        #delete negative one:
        #X = np.delete(X,i,0)
        #W = np.delete(W,i,0)
        #for speed we'll just set W to zero, and delete later...
        W[i] = 0
    X = np.delete(X,np.where(W==0),0)
    W = np.delete(W,np.where(W==0),0)
    return (X,W)



def findbound(X,W,ls,d,gridres,gridstart,gridend,fulldim=False,forceignorenegatives=False,dimthreshold=3):
    """
    X = input locations
    W = 'weights' (i.e. heights) of Gaussians
    ls = lengthscale
    d = number of dimensions
    gridres = number of grid points along longest edge
    gridstart = d-dimensional coordinate of grid start
    gridend = d-dimensional coordinate of grid end
    fulldim = [default false] whether to avoid making the PCA approximation (kicks in if number of dims goes above dimthreshold)
    forceignorenegatives = [default false] we don't need to ignore negatives if we're not using the PCA approximation
    dimthreshold = [default 3] the number of dimensions above which we make the low dimensional approximation.
    
    we have removed the v (variance) parameter, as it is assumed that the heights of the gaussians are equal to the
    values in W, scaling by 'v' is generally not the right thing to do.
    """
    v = 1 #this is set to one, as we don't want to scale the Gaussians again!
    
    assert len(gridstart)==d, "Gridstart & gridend should have same number of items as the number of dimensions (%d)" % d
    
    #if two points are on top of each other, we can combine them.
    #particular useful if either of them is negative, as at high
    #dimensionality we drop the negative points.
    #!For our GPAdversarial bounds method this is already done
    #so it could be left out, but points colocated causes problems
    #with zero-distances in mergenegatives, so I've left it in.
    #
    newX = []
    newW = []
    for x in np.unique(X,axis=0):
        newX.append(x)
        newW.append(np.sum(W[(np.where((X==x).all(axis=1))[0])]))
    X = np.array(newX)
    W = np.array(newW)
    X,W = mergenegatives(X,W,ls)
    
    if X.shape[0]==0: #if no items,
        return 0
    if X.shape[0]==1: #if only one item,
        return max(0,W[0]*v) #todo: not very efficient
    if X.shape[1]>dimthreshold and not fulldim:
        #print("Compacting to %d manifold..." % dimthreshold)
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
    gridspacing = np.max(gridend-gridstart)/gridres
    #TODO should we subtract/add gridspacing to the gridstart/gridend
    return findbound_lowdim(lowdX,W,ls=ls,v=v,d=lowd,gridspacing=gridspacing,gridstart=gridstart,gridend=gridend,ignorenegatives=ignorenegatives)


    
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
