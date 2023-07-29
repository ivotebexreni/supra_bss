import relatesignals as rs
import stats
import numpy as np
from scipy.linalg import inv, sqrtm
# from information import kurtosis

# def filter_sources_f(Y,W,V):
#   n_comp,n_time=Y.shape
#   P=np.zeros((n_comp,n_comp),dtype='bool')
#   # D=W.conj().T.dot(V)
#   K=kurtosis(Y,axis=1)
#   # K=Y.std(axis=1)
#   D=inv(W).dot(V)
#   M=inv(D)
#   Mn=rs.normrows(M)
#   indexp=np.argsort(-K)
#   # for j in range(n_comp):
#   P[indexp,np.arange(n_comp)]=True
#   Z=np.diag(np.diag(M.dot(P))).dot(P.T.dot(Y))
#   return Z,M,P

# def filter_sources(X,
#   algorithm,
#   verbose=False,
#   **kwargs):

#   nc,nf,nt=X.shape
  
#   Z=np.zeros(X.shape,dtype=X.dtype)
#   Xm=np.zeros((nf,nc,1),dtype=X.dtype)
#   W=np.zeros((nf,nc,nc),dtype=X.dtype)
#   V=np.zeros((nf,nc,nc),dtype=X.dtype)
#   M=np.zeros((nf,nc,nc),dtype=X.dtype)

#   # Z=np.zeros(X.shape)
#   # Xm=np.zeros((nf,nc,1))
#   # W=np.zeros((nf,nc,nc))
#   # V=np.zeros((nf,nc,nc))
#   # M=np.zeros((nf,nc,nc))

#   P=np.zeros((nf,nc,nc),dtype='bool')
#   it=np.zeros(nf)
#   t=np.zeros(nf)
#   HW=[]
  
#   for i in range(nf):
#     if verbose: print(i)
#     Y,W[i],V[i],Xm[i],hw,it[i],t[i]=algorithm(X[:,i,:],**kwargs)
#     Z[:,i,:],M[i],P[i]=filter_sources_f(Y,W[i],V[i])
#     HW.append(hw)
#   return Z,W,V,Xm,M,P,HW,it,t

def estimate_sources_f(Y,W,V,ord=1):
  n_comp,n_time=Y.shape
  P=np.zeros((n_comp,n_comp),dtype='bool')
  # D=W.conj().T.dot(V)
  D=inv(W).dot(V)
  # D=inv(W+np.random.normal(0,1e-12,W.shape)).dot(V)
  M=inv(D)
  Mn=rs.normrows(M,ord=ord)
  indexp=rs.argdiag(Mn)
  # print(indexp)
  # for j in range(n_comp):
  P[indexp,np.arange(n_comp)]=True
  Z=np.diag(np.diag(M.dot(P))).dot(P.T.dot(Y))
  return Z,M,P

def estimate_sources(X,
  algorithm,
  verbose=False,
  **kwargs):

  nc,nf,nt=X.shape
  
  Z=np.zeros(X.shape,dtype=X.dtype)
  Xm=np.zeros((nf,nc,1),dtype=X.dtype)
  W=np.zeros((nf,nc,nc),dtype=X.dtype)
  V=np.zeros((nf,nc,nc),dtype=X.dtype)
  M=np.zeros((nf,nc,nc),dtype=X.dtype)

  # Z=np.zeros(X.shape)
  # Xm=np.zeros((nf,nc,1))
  # W=np.zeros((nf,nc,nc))
  # V=np.zeros((nf,nc,nc))
  # M=np.zeros((nf,nc,nc))

  P=np.zeros((nf,nc,nc),dtype='bool')
  it=np.zeros(nf,dtype='int')
  t=np.zeros(nf)
  HW=[]
  
  for i in range(nf):
    if verbose: print(i)
    Y,W[i],V[i],Xm[i],hw,it[i],t[i]=algorithm(X[:,i,:],**kwargs)
    Z[:,i,:],M[i],P[i]=estimate_sources_f(Y,W[i],V[i])
    HW.append(hw)
  return Z,W,V,Xm,M,P,HW,it,t

def estimate_sources_noise_f(Y,W,V,ord=1):
  n_comp,n_time=Y.shape
  P=np.zeros((n_comp,n_comp),dtype='bool')
  # D=W.conj().T.dot(V)
  D=inv(W).dot(V)
  # D=inv(W+np.random.normal(0,1e-12,W.shape)).dot(V)
  M=inv(D)
  noise_comp=np.abs(stats.kurtosis(Y,axis=1)).argmin()
  Mn=rs.normrows(
    M[np.arange(n_comp)!=(n_comp-1)][:,np.arange(n_comp)!=noise_comp],
    ord=ord)
  indexp=np.r_[rs.argdiag(Mn),noise_comp]
  # print(indexp)
  # for j in range(n_comp):
  P[np.r_[indexp,noise_comp],np.arange(n_comp)]=True
  Z=np.diag(np.diag(M.dot(P))).dot(P.T.dot(Y))
  return Z,M,P

def estimate_sources_noise(X,
  algorithm,
  verbose=False,
  **kwargs):

  nc,nf,nt=X.shape
  
  Z=np.zeros(X.shape,dtype=X.dtype)
  Xm=np.zeros((nf,nc,1),dtype=X.dtype)
  W=np.zeros((nf,nc,nc),dtype=X.dtype)
  V=np.zeros((nf,nc,nc),dtype=X.dtype)
  M=np.zeros((nf,nc,nc),dtype=X.dtype)

  # Z=np.zeros(X.shape)
  # Xm=np.zeros((nf,nc,1))
  # W=np.zeros((nf,nc,nc))
  # V=np.zeros((nf,nc,nc))
  # M=np.zeros((nf,nc,nc))

  P=np.zeros((nf,nc,nc),dtype='bool')
  it=np.zeros(nf,dtype='int')
  t=np.zeros(nf)
  HW=[]
  
  for i in range(nf):
    if verbose: print(i)
    Y,W[i],V[i],Xm[i],hw,it[i],t[i]=algorithm(X[:,i,:],**kwargs)
    Z[:,i,:],M[i],P[i]=estimate_sources_f(Y,W[i],V[i])
    HW.append(hw)
  return Z,W,V,Xm,M,P,HW,it,t

def extract_sources_f(X,W,P,V=None,white='zca'):
  Z=X-X.mean(axis=1,keepdims=True)
  if V is None:
    if white=='zca':
      V=inv(sqrtm(Z.dot(Z.conj().T)/Z.shape[1]))
    elif white=='norm':
      V=inv(np.diag(Z.std(axis=1)))
    else:
      V=np.eye(Z.shape[0])
  Z=V.dot(Z)
  # Y=W.conj().T.dot(Z)
  # D=W.conj().T.dot(V)
  Y=inv(W).dot(Z)
  D=inv(W).dot(V)
  M=inv(D)
  Z=np.diag(np.diag(M.dot(P))).dot(P.T.dot(Y))
  return Z

def extract_sources(X,W,P,V=None,white='zca'):
  if V is None:
    V=[None]*W.shape[0]
  # Z=np.zeros(X.shape,dtype='complex')
  Z=np.zeros(X.shape,dtype=X.dtype)
  for i in range(X.shape[1]):
    Z[:,i,:]=extract_sources_f(X[:,i,:],W[i],P[i],V=V[i],white=white)
  return Z

def separate_sources_f(X,W,white='zca'):
  Z=X-X.mean(axis=1,keepdims=True)
  if white=='zca':
    V=inv(sqrtm(Z.dot(Z.conj().T)/Z.shape[1]))
  elif white=='norm':
    V=inv(np.diag(Z.std(axis=1)))
  else:
    V=np.eye(Z.shape[0])
  Z=V.dot(Z)
  # Y=W.conj().T.dot(Z)
  Y=inv(W).dot(Z)
  return Y

def separate_sources(X,W,white='zca'):
  Y=np.zeros(X.shape,dtype=X.dtype)
  for i in range(X.shape[1]):
    Y[:,i,:]=separate_sources_f(X[:,i,:],W[i],white=white)
  return Y