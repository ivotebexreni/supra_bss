import numpy as np

def crosscorr(x1,y1=None):
  y1=y1 if y1 is not None else x1.copy()
  x=x1.copy()
  y=y1.copy()
  x-=x.mean(axis=1,keepdims=True)
  y-=y.mean(axis=1,keepdims=True)
  x/=x.std(axis=1,keepdims=True)
  y/=y.std(axis=1,keepdims=True)
  return x.dot(y.conj().T)/x.shape[1]

def orgmat(C_):
  C=np.abs(C_)
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
  return C_[r_][:,c_]

def orgcrosscorr(x,y):
  return orgmat(crosscorr(x,y))

def orgcrosscorrdiag(x,y):
  return np.diag((orgcrosscorr(x,y)))

def associate(x,y):
  C_=crosscorr(x,y)
  C=np.abs(C_)
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
  return c_[np.argsort(r_)]