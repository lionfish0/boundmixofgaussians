
import numpy as np
from boundmixofgaussians import zeromean_gaussian, findbound, PCA, findbound_lowdim

def test():
    ls = 2.0
    W = 1.0*np.array([3,3])
    approx = findbound(X=1.0*np.array([[0,0,0,0],[1,2,1,1]]),W=W,ls=ls,d=4,gridres=30,gridstart=1.0*np.array([0,0,0,0]),gridend=1.0*np.array([1,1,1,1]),fulldim=True)
    correct = 3*2*np.exp(-.5*(1**2+3*0.5**2)/ls**2)
    assert np.abs(approx-correct)<0.001

    #as there's just two points a 1d manifold will be correct too.
    approx = findbound(X=1.0*np.array([[0,0,0,0],[1,2,1,1]]),W=W,ls=ls,d=4,gridres=10000,gridstart=1.0*np.array([0,0,0,0]),gridend=1.0*np.array([1,1,1,1]),dimthreshold=1)
    assert np.abs(approx-correct)<1e-6

    assert findbound(X=1.0*np.array([[1,2]]),W=1.0*np.array([3]),ls=2.0,d=2,gridres=10,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([2,2]),dimthreshold=3)==3
    
    #assert np.abs(findbound(X=1.0*np.array([[1,2]]),W=1.0*np.array([3]),ls=2.0,d=2,gridres=10,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([2,2]),dimthreshold=3)--1.5908)<0.01
    
    W = 1.0*np.array([1]*7)
    ls = 0.3
    #the biggest value in 2d is a single peak from a single training point
    approx2d = findbound(X=1.0*np.array([[0,0],[1+.5,1-.5],[1-.5,1+.5],[2,2],[3,3],[4,4]]),W=np.array([1,1,0.5,1,1,1]),ls=ls,d=2,gridres=300,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([4,4]),dimthreshold=2)
    assert np.abs(approx2d-1)<0.01

    #when dimensionality is reduced to 1d two peaks cancel out.
    approx1d = findbound(X=1.0*np.array([[0,0],[1+.5,1-.5],[1-.5,1+.5],[2,2],[3,3],[4,4]]),W=np.array([1,1,0.5,1,1,1]),ls=ls,d=2,gridres=300,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([4,4]),dimthreshold=1)
    assert np.abs(approx1d-1.5)<0.01

    #the negative peak shouldn't cancel out the positive one.
    approx1d = findbound(X=1.0*np.array([[0,0],[1+.5,1-.5],[1-.5,1+.5],[2,2],[3,3],[4,4]]),W=np.array([1,2,-1,1,1,1]),ls=ls,d=2,gridres=300,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([4,4]),dimthreshold=1)
    assert np.abs(approx1d-2.0)<0.01

    #the negative peak shouldn't cancel out the positive one if they're just really close (as that's not been coded)
    approx1d = findbound(X=1.0*np.array([[0,0],[1,1],[2,2],[3,3.01],[3,3]]),W=np.array([1,1,1,2,-1]),ls=ls,d=2,gridres=300,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([4,4]),dimthreshold=1)
    approx1daligned = findbound(X=1.0*np.array([[0,0],[1,1],[2,2],[3,3],[3,3]]),W=np.array([1,1,1,2,-1]),ls=ls,d=2,gridres=300,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([4,4]),dimthreshold=1)
    approx2d = findbound(X=1.0*np.array([[0,0],[1,1],[2,2],[3,3.01],[3,3]]),W=np.array([1,1,1,2,-1]),ls=ls,d=2,gridres=300,gridstart=1.0*np.array([0,0]),gridend=1.0*np.array([4,4]),dimthreshold=2)
    assert np.abs(approx2d-1.0)<0.01 #negative cancels out +ve peak (2-1) in 2d
    assert np.abs(approx1daligned-1.0)<0.01 #negative will cancel out +ve peak if correctly aligned (in 1d)
    assert np.abs(approx1d-2.0)<0.01 #negative ignored in 1d
    
    #test the code that combines negatives into positive peaks
    from boundmixofgaussians import zeromean_gaussian_1d, compute_grad, compute_sum, mergenegatives
    assert np.abs(((compute_sum(3,-1,0.1,1.2,1.2)-compute_sum(3,-1,0.1+0.0000001,1.2,1.2))/0.0000001)-compute_grad(3,-1,0.1,1.2,1.2))<0.00001
    #if ls is large relative to distances, then the solution
    #to the gradient=0 is at A.x = B.x-B.dist (A-B)x = dist*B/(A-B)
    #in the case below dist = 0.02236068

    dist = 0.02236068
    X=1.0*np.array([[0,0],[1,1],[2,2],[3.02,3.01],[3,3]])
    W=1.0*np.array([1,1,1,4,-2])
    #here A=4, B=2: 2/(4-2)=2/2=1
    newX,newW = mergenegatives(X.copy(),W,10.0)
    assert (newX[3,1]-(dist*1+X[3,1])<1e-5)


    X=1.0*np.array([[0,0],[1,1],[2,2],[3,3.01],[3,3]])
    W=1.0*np.array([1,1,1,5,-1])
    #here A=5, B=1: 1/(5-1)=1/4
    newX,newW = mergenegatives(X.copy(),W,5.0)
    assert (newX[3,1]-(dist*(1/4)+X[3,1])<1e-5)
    #note that the lengthscale doesn't come into the calc.

    #Test that a distant negative will have no impact
    X=1.0*np.array([[0,0],[1,1],[2,2],[3,103],[3,3]])
    W=1.0*np.array([1,1,1,-2,4])
    newX,newW = mergenegatives(X.copy(),W,2.0)
    assert (np.all(newW==np.array([1,1,1,4])))
    
    #Test the find peak method correctly evaluates the peak
    from boundmixofgaussians import zeromean_gaussian_1d as zmg
    from boundmixofgaussians import findpeak
    x, p = findpeak(0.5,-1.5,2.0,1.0)
    assert zmg(x,1.0,0.5)+zmg(x-2.0,1.0,-1.5)==p
    #confirm we've found the peak
    for delta in [0.001,0.01,-0.001,-0.01]:
        assert p>(zmg(x+delta,1.0,0.5)+zmg(x-2.0+delta,1.0,-1.5))
