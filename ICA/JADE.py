import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, inv, eig
import time

def JADE(
  X,
  n_components=None,
  max_iter=1000,
  tol=1e-4,
  whiten=True,
  W_init=None,
  random_state=None):

  tic=time.time()

  random_state=random_state if np.random.randint(65536) is None else random_state
  rng=np.random.default_rng(random_state)

  n,t=X.shape
  if n_components is not None:
    n=n_components

  xm=X.mean(axis=1,keepdims=True)
  z=X-xm

  if whiten:
    V=inv(sqrtm(z.dot(z.conj().T)/t))
    
    # E,L=eig(z.dot(z.conj().T)/t)
    # V=np.sqrt(inv(np.diag(E))).dot(L.conj().T)[:n]
    # z=V.dot(z)

  else:
    V=np.eye(n)

  z=V.dot(z)
  # perm=np.random.choice(t,int(t*n_train),replace=False)
  # z=u[:,perm]

  HW=np.ones((max_iter,n,n),dtype='complex')*np.nan
  na=int(n*(n-1)/2)
  HM=np.ones((max_iter,na,2),dtype='complex')*np.nan

  Q=np.zeros((n*n,n*n),dtype='complex')

  for i in range(n):
    for j in range(n):
      for k in range(n):
        for l in range(n):
          Q[n*i+j,n*k+l]=(z[i]*z[j].conj()*z[k].conj()*z[l]).mean()-(z[i]*z[j].conj()).mean()*(z[k]*z[l].conj()).mean()-(z[i]*z[l]).mean()*(z[j].conj()*z[k].conj()).mean()-(z[i]*z[k].conj()).mean()*(z[l].conj()*z[j]).mean()

  E,I=eig(Q)
  E=np.abs(E)
  Eind=np.argsort(E)
  E=E[Eind]

  nm=n
  M=np.zeros((nm,n,n),dtype='complex')
  for i in range(nm):
    Z=I[:,Eind[n*n-i-1]].reshape(n,n)
    Z/=np.exp(1j*np.angle(Z[0,0]))
    M[i,:,:]=Z*E[n*n-i-1]

  # W=W_init if W_init is not None else rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
  # W=W.dot(sqrtm(inv(W.conj().T.dot(W))))

  W=np.eye(n,dtype='complex')

  n_=np.arange(n-1,0,-1)

  D=np.array([[1,0,0,-1],[0,1,1,0],[0,-1j,1j,0]])

  for it in range(max_iter):
    for i in range(n-1):
      for j in range(i+1,n):
        G=np.zeros((3,3),dtype='complex')
        ind=np.r_[i,j]
        for k in range(nm):
          h=D.dot((M[k,ind][:,ind]).flatten())
          G+=np.outer(h.conj().T,h)
        G=np.real(G)
        le,li=eig(G)
        lx=li[:,np.argsort(np.abs(le))[2]]
        lx=-lx if lx[0]<0 else lx
        x_,y_,z_=tuple(lx)
        c=np.sqrt((x_+1)/2)
        s=(y_-1j*z_)/np.sqrt(2*(x_+1))
        H=np.eye(n,dtype='complex')
        H[i,ind]=np.r_[c,-np.conj(s)]
        H[j,ind]=np.r_[s,c]
        W=W.dot(H)
        HM[it,n_[:i+1].sum()-(n-j),:]=np.array([c,s])
        for k in range(nm):
          M[k]=H.conj().T.dot(M[k]).dot(H)
    HW[it]=W
    if np.abs(np.abs(HM[it])-np.c_[np.ones(na),np.zeros(na)]).mean()<=tol:
      break

  if (i+1)>=max_iter and np.abs(np.abs(HM[it])-np.c_[np.ones(na),np.zeros(na)]).mean()>tol:
    warnings.warn('Maximum number of iterations reached. ICA did not converge.')

  Y=W.conj().T.dot(z)

  HW=HW[np.logical_not(np.isnan(HW))].reshape((it+1,n,n))
  HM=HM[np.logical_not(np.isnan(HM))].reshape((it+1,int(n*(n-1)/2),2))
  
  toc=time.time()

  return  Y,W,V,xm,HW,it+1,toc-tic