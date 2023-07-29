import numpy as np
import warnings
from scipy.linalg import sqrtm, inv, eig, norm
import time

def Kurt(
  x,
  algorithm='parallel',
  n_components=None,
  max_iter=1000,
  tol=1e-4,
  whiten=True,
  W_init=None,
  random_state=None):
  
  tic=time.time()
  
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
      Y=W.conj().T.dot(z)
      W=(2*(Y*Y.conj()*Y.conj()).dot(z.T)/t-4*(Y.conj().dot(z.T))/t-2*np.diag(np.diag((Y.conj().dot(Y.conj().T))/t)).dot(Y.dot(z.T))/t).T
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