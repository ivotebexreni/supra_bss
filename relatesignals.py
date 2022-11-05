import numpy as np

def normrows(A):
  return np.abs(A)/np.linalg.norm(np.abs(A),ord=1,axis=1,keepdims=True)

def argdiag(C_):
  C=C_.copy()
  n=C.shape[0]
  r=np.arange(n)
  c=np.arange(n)
  r_=np.zeros(n,dtype='int')
  c_=np.zeros(n,dtype='int')
  for i in range(n):
    ir=int(np.argmax(np.max(C[r][:,c].reshape((n-i,n-i)),axis=1)))
    ic=int(np.argmax(np.max(C[r][:,c].reshape((n-i,n-i)),axis=0)))
    r_[i]=r[ir]
    c_[i]=c[ic]
    r=np.delete(r,np.where(r==r_[i]))
    c=np.delete(c,np.where(c==c_[i]))
  s_=np.argsort(r_)
  return c_[s_]
