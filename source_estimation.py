# import CICA_opt as cica
import CICA as cica
import relatesignals as rs
import numpy as np

def estimate_sources_FCICA(X,random_state=int(np.random.rand()*1000)):
  n_comp,n_freq,n_time=X.shape
  Y=np.zeros(X.shape,dtype='complex')
  S=np.zeros(X.shape,dtype='complex')
  W=np.zeros((n_freq,n_comp,n_comp),dtype='complex')
  V=np.zeros((n_freq,n_comp,n_comp),dtype='complex')
  D=np.zeros((n_freq,n_comp,n_comp),dtype='complex')
  M=np.zeros((n_freq,n_comp,n_comp),dtype='complex')
  Xm=np.zeros((n_freq,n_comp),dtype='complex')
  P=np.zeros((n_freq,n_comp,n_comp),dtype='bool')
  
  HGf=[]
  HWf=[]
  HDf=[]
  
  n_iter=np.zeros((n_freq,))
  time=np.zeros((n_freq,))
  nFreq=n_freq
  for i in range(nFreq):
    S[:,i,:],w,v,xm,hg,hw,hd,n_iter[i],time[i]=cica.FastCICA(X[:,i,:],algorithm='parallel',n_components=None,
    max_iter=10000,tol=1e-5,whiten=True,contrast=None,a=.1,W_init=None,
    random_state=random_state)
    W[i,:,:]=w
    V[i,:,:]=v
    Xm[i,:]=xm.flatten()
    HGf.append(hg)
    HWf.append(hw)
    HDf.append(hd)

    D[i,:,:]=W[i,:,:].conj().T.dot(V[i,:,:])
    M[i,:,:]=np.linalg.inv(D[i,:,:])
    Mn=rs.normrows(M[i,:,:])
    indexp=rs.argdiag(Mn)
    for j in range(n_comp):
      P[i,indexp[j],j]=True
    Y[np.arange(5),i,:]=np.diag(np.diag(M[i,:,:].dot(P[i,:,:]))).dot(P[i,:,:].T.dot(S[:,i,:]))
    print('Frequency:',i,'kHz')
  return Y,W,V,M,P,Xm,HGf,HWf,HDf,n_iter,time
