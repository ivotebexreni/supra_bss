import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

nq=1/2*(1+erf(1/np.sqrt(2)))
nql=.5-nq/1
nqu=.5+nq/1

def complexnormal(n=None,seed=None):
  seed=np.random.randint(65536) if seed is None else seed
  rng=np.random.default_rng(seed)
  return (rng.normal(0,1,n)+1j*rng.normal(0,1,n))/np.sqrt(2)

def complexuniform(n=None,seed=None):
  seed=np.random.randint(65536) if seed is None else seed
  rng=np.random.default_rng(seed)
  return (rng.uniform(0,1,n)+1j*rng.uniform(0,1,n))/np.sqrt(2)

def complexlaplace(n=None,seed=None):
  seed=np.random.randint(65536) if seed is None else seed
  rng=np.random.default_rng(seed)
  return (rng.laplace(0,1,n)+1j*rng.laplace(0,1,n))/np.sqrt(2)

def complexexponential(n=None,seed=None):
  seed=np.random.randint(65536) if seed is None else seed
  rng=np.random.default_rng(seed)
  return (rng.exponential(1,n)+1j*rng.exponential(1,n))/np.sqrt(2)

def uncertainty(x,q=None,axis=-1):
  q = nq if q is None else q
  q = q if q==.5 else q%.5
  ul=np.quantile(x,q  ,axis=axis)
  um=np.quantile(x,.5 ,axis=axis)
  uh=np.quantile(x,1-q,axis=axis)
  return um-ul,uh-um

def stdvar(v,axis=None):
  v-=v.mean(axis=axis,keepdims=True)
  v/=v.std(axis=axis,keepdims=True)

def hist(x,histtype='step',**kwargs):
  return plt.hist(x,histtype=histtype,**kwargs)

def errorbar(x,y,q=nq,plot=True,**kwargs):
  if type(q)==type('str'):
    p=y.mean(axis=1)
    el=y.std(axis=1)
    et=y.std(axis=1)
  else:
    p=np.median(y,axis=1)
    qd=np.min([np.abs(1-q),np.abs(q)])
    el=p-np.quantile(y,qd,axis=1)
    et=np.quantile(y,1-qd,axis=1)-p
  if plot:
    eb=plt.errorbar(x,p,yerr=np.c_[el,et].T,**kwargs)
    return eb,(p,el,et)
  else:
    return p,el,et

def fillerrorbar(x,y,q=nq,plot=True,lw=1,lc='tab:blue',ls='-',fc='tab:blue',alpha=.5):
  if type(q)==type('str'):
    p=y.mean(axis=1)
    el=y.std(axis=1)
    et=y.std(axis=1)
  else:
    p=np.median(y,axis=1)
    qd=np.min([np.abs(1-q),np.abs(q)])
    el=p-np.quantile(y,qd,axis=1)
    et=np.quantile(y,1-qd,axis=1)-p
    
  plt.fill_between(x,p-el,p+et,fc=fc,alpha=alpha)
  plt.plot(x,p,lw=lw,color=lc,ls=ls)

def getfrequencyindex(w,fs=500e3,flim=(2000,150000)):
  f=np.fft.fftfreq(w,1/fs)
  return np.where(np.logical_and(f>=flim[0],f<=flim[1]))[0]