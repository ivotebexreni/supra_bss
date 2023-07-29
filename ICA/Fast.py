import numpy as np
import warnings
from scipy.linalg import sqrtm, inv, eig, norm
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
  return -2*g1(y,a)**3

def G2(y,a):
  return np.log(a+y)
def g2(y,a):
  return 1/(y+a)
def dg2(y,a):
  return -g2(y,a)**2

def Fast(
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

  random_state=np.random.randint(65536) if random_state is None else random_state
  rng=np.random.default_rng(random_state)

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

  HW=np.ones((max_iter,n,n),dtype='complex')*np.nan
  HD=np.ones((max_iter,),dtype='float')*np.nan

  # if algorithm=='deflation':
  #   W=np.zeros((n,n),dtype=np.complex)

  #   for k in range(n):
  #     w=W_init[:,k] if W_init is not None else rng.normal(size=(n,))+1j*rng.normal(size=(n,))
  #     w/=norm(w)

  #     for i in range(max_iter):
  #       w_old=w.copy()
  #       sab=sqabs(w,z)
  #       gw=g(sab)
  #       dgw=dg(sab)
  #       w=(z*(w.conj().T.dot(z)).conj()*gw).mean(axis=1)-(gw+sab*dgw).mean()*w
  #       w/=norm(w)
  #       for j in range(k):
  #         w-=(W[:,k].conj().T.dot(w)*W[:,k])
  #         w/=norm(w)
  #       W[:,k]=w.copy()

  #       HW[i,:,k]=w.copy()

  #       if norm(np.imag(np.diag(w.dot(w_old.conj()))))<tol:
  #         break
  #     if norm(np.imag(np.diag(HW[i].dot(HW[i-1].conj().T))))>tol:
  #       warnings.warn('FastCICA did not converge. Consider increasing tolerance or the maximum number of iterations.')

  if algorithm=='parallel':
    W=W_init if W_init is not None else rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    W=W.dot(sqrtm(inv(W.conj().T.dot(W))))

    for i in range(max_iter):
      W_old=W.copy()
      Sab=sqabs(W,z)
      gW=g(Sab)
      dgW=dg(Sab)
      W=(z.dot((W.conj().T.dot(z).conj()*gW).T))/t-((gW+Sab*dgW).mean(axis=1)*W)
      W=W.dot(sqrtm(inv(W.conj().T.dot(W))))
      HW[i,:,:]=W.copy()
      HD[i]=norm(np.imag(np.diag(W.dot(W_old.conj().T))))
      
      if HD[i]<tol:
        break
      if i>12 and np.abs((np.diff(HD[:i])[-12:]).sum())<tol:
        break
    if (i+1)>=max_iter:
      if norm(np.imag(np.diag(HW[i].dot(HW[i-1].conj().T))))>tol:
        warnings.warn('Maximum number of iterations reached. ICA did not converge.')

  HW=HW[np.logical_not(np.isnan(HW))].reshape((i+1,n,n))
  HD=HD[np.logical_not(np.isnan(HD))]
  Y=W.conj().T.dot(z)

  toc=time.time()
    
  return  Y,W,V,xm,(HW,HD),i+1,toc-tic