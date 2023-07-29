import numpy as np
from scipy.stats import entropy, gaussian_kde as kde
from scipy.stats import gaussian_kde
from scipy.signal import coherence
# from scipy.interpolate
# from dcor import distance_correlation
from util import stdvar
import warnings

def gkldiv(x,y,axis=None):
  axis=axis if axis is not None else tuple(np.arange(x.ndim))
  return entropy(np.abs(x),np.abs(y),axis=axis)

def gkldivhtest(x_,y_,n=1000,m=1000,d=1000,seedx=None,seedy=None):
  return hipotest(x_,y_,gkldiv,1000,1000,1000,None,None)

# def dcor(x_,y_):
#   x=np.abs(x_)
#   y=np.abs(y_)
#   if len(x.shape)==1:
#     return distance_correlation(x,y)
#   else:
#     r=np.zeros(x.shape[1])
#     for i in range(x.shape[1]):
#       r[i]=distance_correlation(x[:,i],y[:,i])
#     return r
  
def hipotest(x_,y_,f,n=1000,m=1000,d=1000,seedx=None,seedy=None):
  x=np.abs(x_)
  y=np.abs(y_)
  kx=kde(x)
  ky=kde(y)
  vx=kx.resample(n*m,seedx).reshape(n,m)
  vy=ky.resample(n*m,seedy).reshape(n,m)
  kld=f(x,y)
  vkld=f(vx,vy)
  kvkld=kde(vkld)
  t=np.linspace(0,kld,d+1)
  px=kvkld.pdf(t)
  return ((px[:-1]+px[1:])*np.diff(t)/2).sum(),kld

def kldiv(x_,y_,n=200,std=True):
  x=x_.copy()
  y=y_.copy()
  if std:
    stdvar(x)
    stdvar(y)
  kx=kde(x)
  ky=kde(y)
  t=np.linspace(np.r_[x,y].min()-.1,np.r_[x,y].max()+.1,n+1)
  p=kx.pdf(t)
  q=ky.pdf(t)
  kl=np.zeros(p.shape)
  kl[np.logical_and(p>0,q>0)]=p[np.logical_and(p>0,q>0)]*np.log(p[np.logical_and(p>0,q>0)]/q[np.logical_and(p>0,q>0)])
  kl[np.logical_and(p==0,q>=0)]=q[np.logical_and(p==0,q>=0)]
  # kl[np.logical_and(p>0,q==0)]=np.inf
  return ((kl[:-1]+kl[1:])*np.diff(t)/2).sum()

def diff_entropy(x_,l=None,d=0,n=200,std=True):
  x=x_.copy()
  if std:
    stdvar(x)
  kx=kde(x)
  l=l if l is not None else (x.min(),x.max())
  t=np.linspace(l[0]-d,l[1]+d,n+1)
  p=kx.pdf(t)
  p=p[p>0]
  en=-p*np.log(p)
  return ((en[1:]+en[:-1])*np.diff(t)/2).sum()

def diff_entropy_hist(x_,h=None,std=True):
  x=x_.flatten()
  if std:
    stdvar(x)
  h=h if h is not None else x.std()*(24*np.sqrt(np.pi)/x.size)**(1/3)
  t=np.arange(x.min(),x.max()+h,h)
  p=np.histogram(x,bins=t,density=True)[0]
  p=p[p>0]
  en=-p*np.log(p)*h
  return en.sum()

def joint_entropy(x_,y_,l=[None,None],d=(0,0),n=(200,200),std=True):
  x=x_.copy()
  y=y_.copy()
  if std:
    stdvar(x)
    stdvar(y)
  try:
    k=kde(np.c_[x,y].T)
  except np.linalg.LinAlgError as e:
    return diff_entropy(x)
  l[0]=l[0] if l[0] is not None else (x.min(),x.max())
  l[1]=l[1] if l[1] is not None else (y.min(),y.max())
  tx,ty=np.meshgrid(np.linspace(l[0][0],l[0][1],n[0]),np.linspace(l[1][0],l[1][1],n[1]))
  t=np.c_[tx.flatten(),ty.flatten()].T
  pxy=k.pdf(t).reshape(n[1],n[0])
  en_index=pxy>0
  en=pxy[en_index]*np.log(pxy[en_index])*(tx[0,1]-tx[0,0])*(ty[1,0]-ty[0,0])
  return -en.sum()

def joint_entropy_hist(x_,y_,h=[None,None],std=True):
  x=x_.flatten()
  y=y_.flatten()
  if std:
    stdvar(x)
    stdvar(y)
  h[0]=h[0] if h[0] is not None else (x.max()-x.min())/((x.max()-x.min())/(x.std()*(24*np.sqrt(np.pi)/x.size)**(1/3)))**(1/2)
  h[1]=h[1] if h[1] is not None else (y.max()-y.min())/((y.max()-y.min())/(y.std()*(24*np.sqrt(np.pi)/y.size)**(1/3)))**(1/2)
  # print(h)
  # h=[x.std(),y.std()]
  t=[np.arange(x.min(),x.max()+h[0],h[0]),np.arange(y.min(),y.max()+h[1],h[1])]
  pxy=np.histogram2d(x,y,bins=t,density=True)[0]
  en_index=pxy>0
  en=pxy[en_index]*np.log(pxy[en_index])*h[0]*h[1]
  return -en.sum()

def mutual_info(x_,y_,l=[None,None],d=(0,0),n=(200,200),std=True):
  enx=diff_entropy(x_,l=l[0],d=d[0],n=n[0],std=std)
  eny=diff_entropy(y_,l=l[1],d=d[1],n=n[1],std=std)
  en_=joint_entropy(x_,y_,l=l,d=(0,0),n=n,std=True)
  return enx+eny-en_

def mutual_info_hist(x_,y_,h=[None,None],std=True):
  enx=diff_entropy_hist(x_,h=h[0],std=std)
  eny=diff_entropy_hist(y_,h=h[1],std=std)
  en_=joint_entropy_hist(x_,y_,h=h,std=std)
  return enx+eny-en_

# def mutual_info(x_,y_,l=None,d=(0,0),n=(200,200),std=True):
#   x=x_.copy()
#   y=y_.copy()
#   if std:
#     stdvar(x)
#     stdvar(y)
#   try:
#     k=kde(np.c_[x,y].T)
#   except np.linalg.LinAlgError as e:
#     return diff_entropy(x)
#   l=l if l is not None else (x.min(),x.max(),y.min(),y.max())
#   tx,ty=np.meshgrid(np.linspace(l[0],l[1],n[0]),np.linspace(l[2],l[3],n[1]))
#   t=np.c_[tx.flatten(),ty.flatten()].T
#   pxy=k.pdf(t).reshape(n[1],n[0])
#   px=((pxy[:-1,:]+pxy[1:,:])*(ty[1,0]-ty[0,0])/2).sum(axis=0)
#   py=((pxy[:,:-1]+pxy[:,1:])*(tx[0,1]-tx[0,0])/2).sum(axis=1)
#   px_=np.tile(px,(n[1],1))
#   py_=np.tile(py,(n[0],1)).T
#   mi=np.zeros(pxy.shape)
#   mi_index=np.logical_and(pxy>0,np.logical_and(px_>0,py_>0))
#   mi=pxy[mi_index]*np.log(pxy[mi_index]/px_[mi_index]/py_[mi_index])*(tx[0,1]-tx[0,0])*(ty[1,0]-ty[0,0])
#   return mi.sum()

# def mutual_info_hist(x_,y_,h=None,d=(0,0),std=True):
#   x=x_.flatten()
#   y=y_.flatten()
#   if std:
#     stdvar(x)
#     stdvar(y)
#   h=h if h is not None else np.array([x.std()*(24*np.sqrt(np.pi)/x.size),y.std()*(24*np.sqrt(np.pi)/y.size)])**(1/3)
#   t=[np.arange(x.min(),x.max()+h[0],h[0]),np.arange(y.min(),y.max()+h[1],h[1])]
#   # print((t[0].size-1,t[1].size-1))
#   pxy=np.histogram2d(x,y,bins=t,density=True)[0]
#   # print(pxy.shape)
#   px=((pxy)*h[1]).sum(axis=1)
#   py=((pxy)*h[0]).sum(axis=0)
#   # plt.plot(px)
#   # plt.plot(py)
#   # print((px.size,py.size))
#   px_=np.tile(px,(py.size,1)).T
#   # print(px_.shape)
#   py_=np.tile(py,(px.size,1))
#   # print(py_.shape)
#   # mi=np.zeros(pxy.shape)
#   mi_index=np.logical_and(pxy>0,np.logical_and(px_>0,py_>0))
#   mi=pxy[mi_index]*np.log(pxy[mi_index]/px_[mi_index]/py_[mi_index])*h[0]*h[1]
#   return mi.sum()

# def mutual_info0(x_,y_,l=None,d=(0,0),n=(200,200),std=True):
#   x=x_.copy()
#   y=y_.copy()
#   if std:
#     stdvar(x)
#     stdvar(y)
#   try:
#     k=kde(np.c_[x,y].T)
#   except np.linalg.LinAlgError as e:
#     return diff_entropy(x)
#   l=l if l is not None else (x.min(),x.max(),y.min(),y.max())
#   tx,ty=np.meshgrid(np.linspace(l[0],l[1],n[0]+1),np.linspace(l[2],l[3],n[1]+1))
#   t=np.c_[tx.flatten(),ty.flatten()].T
#   pxy=k.pdf(t).reshape(n[1]+1,n[0]+1)
#   px=np.expand_dims(pxy.sum(axis=0)*(ty[1,0]-ty[0,0]),axis=0)
#   py=np.expand_dims(pxy.sum(axis=1)*(tx[0,1]-tx[0,0]),axis=1)
#   warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
#   warnings.filterwarnings("ignore", message="divide by zero encountered in log")
#   warnings.filterwarnings("ignore", message="invalid value encountered in multiply")
#   warnings.filterwarnings("ignore", message="invalid value encountered in log")
#   mi=pxy*(tx[0,1]-tx[0,0])*(ty[1,0]-ty[0,0])*np.log(pxy/px/py)
#   mi=mi[np.logical_not(np.isnan(mi))]
#   mi=mi[np.logical_not(np.isinf(mi))]
#   warnings.filterwarnings("default", message="invalid value encountered in true_divide")
#   warnings.filterwarnings("default", message="divide by zero encountered in log")
#   warnings.filterwarnings("default", message="invalid value encountered in multiply")
#   warnings.filterwarnings("default", message="invalid value encountered in log")
#   return mi.sum()

def kurtosis(x_,axis=None):
  axis=axis if axis is not None else tuple(np.arange(x_.ndim))
  x=x_.copy();x-=x.mean(axis=axis,keepdims=True);x/=x.std(axis=axis,keepdims=True);xc=x.conj()
  k=np.real(\
    (x*xc*x*xc).mean(axis=axis)-\
    (x*xc).mean(axis=axis)**2*2-\
    (x*x).mean(axis=axis)*(xc*xc).mean(axis=axis))
  return k

def cumulant(x_,y_,axis=None):
  axis=axis if axis is not None else tuple(np.arange(x_.ndim))
  x=x_.copy();x-=x.mean(axis=axis,keepdims=True);x/=x.std(axis=axis,keepdims=True);xc=x.conj()
  y=y_.copy();y-=y.mean(axis=axis,keepdims=True);y/=y.std(axis=axis,keepdims=True);yc=y.conj()
  return np.real(
         (x*xc*yc*y).mean(axis=axis)\
        -( x*xc).mean(axis=axis)*( y*yc).mean(axis=axis)\
        -( x*y ).mean(axis=axis)*(xc*yc).mean(axis=axis)\
        -( x*yc).mean(axis=axis)*(xc*y ).mean(axis=axis))

def norm_cumulant(x_,y_,axis=None):
  return cumulant(x_,y_,axis=axis)/np.sqrt(np.abs(kurtosis(x_,axis=axis)*kurtosis(y_,axis=axis)))

def sir(B_,Y_,dB=True,axis=None):
  axis=axis if axis is not None else tuple(np.arange(B_.ndim))
  B=np.abs(B_)
  Y=np.abs(Y_)
  D=Y-B
  s=(B**2).sum(axis=axis)/(D**2).sum(axis=axis)
  s=10*np.log10(s) if dB else s
  return s

def correlation(x_,y_,axis=None):
  axis=axis if axis is not None else tuple(np.arange(x_.ndim))
  x=x_.copy();x-=x.mean(axis=axis,keepdims=True);x/=x.std(axis=axis,keepdims=True)
  y=y_.copy();y-=y.mean(axis=axis,keepdims=True);y/=y.std(axis=axis,keepdims=True)
  return (x*y.conj()).mean(axis=axis)

def spindex(c_,axis=None):
  c=np.array(c_)
  axis=axis if axis is not None else tuple(np.arange(c.ndim))
  n=np.prod(np.array(c.shape)[np.array(axis)])
  # sp=np.sqrt((c.sum(axis=axis)/n)*(c.prod(axis=axis)**(1/n)))
  sp=np.sqrt((c.sum(axis=axis)/n)*((c**(1/n)).prod(axis=axis)))
  return sp