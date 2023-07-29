import numpy as np
import warnings
from scipy.linalg import sqrtm, inv, eig, norm
import time

def FOBI(
  x,
  n_components=None,
  max_iter=None,
  tol=None,
  whiten=True,
  random_state=None):
  
  tic=time.time()

  n,t=x.shape
  if n_components is not None:
    n=n_components

  xm=x.mean(axis=1,keepdims=True)
  z=x-xm

  if whiten:    
    V=inv(sqrtm(z.dot(z.conj().T)/t))
  else:
    V=np.eye(n)
  
  z=V.dot(z)

  HW=np.ones((1,n,n),dtype='complex')*np.nan
  HD=np.ones((1,),dtype='float')*np.nan

  z_=z*norm(z,axis=0,keepdims=True)
  Cz=z_.dot(z_.conj().T)/t
  E,W=eig(Cz)
  # W=L.dot(np.sqrt(np.diag(1/E.conj())))
  # W=L

  HW[0,:,:]=W
  HD[0]=0

  Y=W.conj().T.dot(z)

  toc=time.time()
    
  return  Y,W,V,xm,(HW,HD),1,toc-tic