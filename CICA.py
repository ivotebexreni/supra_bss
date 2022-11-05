import numpy as np
import warnings
from scipy.linalg import sqrtm
import time

def sqabs(W,X):
  return np.abs(W.conj().T.dot(X))**2

def G0(y):
  return (y**2)/2
def g0(y):
  return y
def dg0(y):
  return np.ones(y.shape)

def G1(y,a):
  return np.sqrt(a+y)
def g1(y,a):
  return 1/2/G1(y,a)
def dg1(y,a):
  return 2*g1(y,a)**3

def G2(y,a):
  return np.log(a+y)
def g2(y,a):
  return 1/(y+a)
def dg2(y,a):
  return -g2(y,a)**2

def FastCICA(
  x,
  algorithm='parallel',
  n_components=None,
  max_iter=1000,
  tol=1e-4,
  whiten=True,
  contrast=None,
  a=.1,
  W_init=None,
  random_state=None):
  
  tic=time.time()
  
  if contrast=='sq':
    G = lambda x: G0(x)
    g = lambda x: g0(x)
    dg= lambda x:dg0(x)
  elif contrast=='sqrt':
    G = lambda x: G1(x,a)
    g = lambda x: g1(x,a)
    dg= lambda x:dg1(x,a)
  else:
    G = lambda x: G2(x,a)
    g = lambda x: g2(x,a)
    dg= lambda x:dg2(x,a)

  random_state=random_state if np.random.randint(65536) is None else random_state
  rng=np.random.default_rng(random_state)
             
  n_comp,n_states=x.shape
  if n_components is not None:
    n_comp=n_components

  if whiten:
    xm=x.mean(axis=1,keepdims=True)
    z=x-xm
    
    E,L=np.linalg.eig(z.dot(z.conj().T)/n_states)
    V=np.sqrt(np.linalg.inv(np.diag(E))).dot(L.conj().T)[:n_comp]
    z=V.dot(z)
  else:
    V=np.eye(n_comp)

  HG=np.ones((max_iter,n_comp))*np.nan
  HW=np.ones((max_iter,n_comp,n_comp),dtype='complex')*np.nan
  HD=np.ones((max_iter,n_comp))*np.nan

  if algorithm=='deflation':
    W=np.zeros((n_comp,n_comp),dtype=np.complex)

    for k in range(n_comp):
      w=W_init[:,k] if W_init is not None else rng.normal(size=(n_comp,))+1j*rng.normal(size=(n_comp,))
      w/=np.linalg.norm(w)

      for i in range(max_iter):
        w_old=w.copy()
        sab=sqabs(w,z)
        gw=g(sab)
        dgw=dg(sab)
        w=(z*(w.conj().T.dot(z)).conj()*gw).mean(axis=1)-(gw+sab*dgw).mean()*w
        w/=np.linalg.norm(w)
        for j in range(k):
          w-=(W[:,k].conj().T.dot(w)*W[:,k])
          w/=np.linalg.norm(w)
        W[:,k]=w.copy()

        HG[i,k]=G(sqabs(w,z)).mean()
        HW[i,:,k]=w.copy()
        HD[i,k]=np.abs(w.dot(w_old.conj().T))

        if np.abs(HD[i,k]-1)<tol:
          break
      if (i+1)>=max_iter and (HD[i,k]-1)>tol:
        warnings.warn('FastCICA did not converge. Consider increasing tolerance or the maximum number of iterations.')

  if algorithm=='parallel':
    W=W_init if W_init is not None else rng.normal(size=(n_comp,n_comp))+1j*rng.normal(size=(n_comp,n_comp))
    W=W.dot(sqrtm(np.linalg.inv(W.conj().T.dot(W))))
    
    for i in range(max_iter):
      W_old=W.copy()
      for k in range(n_comp):
        w=W[:,k].copy()
        sab=sqabs(w,z)
        gw=g(sab)
        dgw=dg(sab)
        w=(z*(w.conj().T.dot(z)).conj()*gw).mean(axis=1)-(gw+sab*dgw).mean()*w
        W[:,k]=w.copy()
      W=W.dot(sqrtm(np.linalg.inv(W.conj().T.dot(W))))
      # print(W)
      
      HG[i,:]=G(sqabs(W,z)).mean(axis=1)
      HW[i,:,:]=W.copy()
      HD[i,:]=np.diag(np.abs(W.dot(W_old.conj().T)))
      # print(W)
      # print(W_old)

      if (np.abs(np.abs(HD[i,:])-1)).mean()<tol:
        break
    if (i+1)>=max_iter and (np.abs(HD[i,:]-1)).mean()>tol:
      warnings.warn('FastCICA did not converge. Consider increasing tolerance or the maximum number of iterations.')

  HD=HD[np.logical_not(np.isnan(HD))].reshape((i+1,n_comp))
  HW=HW[np.logical_not(np.isnan(HW))].reshape((i+1,n_comp,n_comp))
  HG=HG[np.logical_not(np.isnan(HG))].reshape((i+1,n_comp))
  # HG=HG[np.logical_not(np.isnan(HG))]
  # HW=HW[np.logical_not(np.isnan(HW))]
  # HD=HD[np.logical_not(np.isnan(HD))]
  Y=W.conj().T.dot(z)

  toc=time.time()
    
  return  Y,W,V,xm,HG,HW,HD,i+1,toc-tic