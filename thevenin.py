import numpy as np

def series(*zs):
  return np.array(zs).sum()

def parallel(*zs):
  return 1/(1/np.array(zs)).sum()

def thevenineq_grid(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n_):
  n=np.array(n_)
  Z=np.inf
  for i,l in enumerate(set(n)):
    Z_=np.inf
    for e in np.where(n==l)[0]:
      Z_=parallel(Z_,series(Zpn[e],Zsn[e]))
    Z_=series(Z_,Zl[i])
    Z=parallel(Z,Z_)
  Z=parallel(Z,Zg)
  Z=series(Z,Zs0)
  return Z

def thevenineq_equip(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n_,eqi_):
  eqi=eqi_-1
  n=np.array(n_)
  Z=np.inf
  for i,l in enumerate(set(n)):
    if l!=n[eqi]:
      Z_=np.inf
      for e in np.where(n==l)[0]:
        Z_=parallel(Z_,series(Zpn[e],Zsn[e]))
      Z_=series(Z_,Zl[i])
      Z=parallel(Z,Z_)
  Z=parallel(Z,series(Zs0,Zp0))
  Z=parallel(Z,Zg)
  Z=series(Z,Zl[n[eqi]-2])
  for i in range(len(n)):
    if n[i]==n[eqi] and i!=eqi:
      Z=parallel(Z,series(Zpn[i],Zsn[i]))
  Z=series(Z,Zsn[eqi])
  return Z

def thevenineq(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n):
  Z=np.zeros((len(n)+1,),dtype='complex')
  Z[0]=thevenineq_grid(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n)
  for i in range(1,Z.shape[0]):
    Z[i]=thevenineq_equip(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n,i)
  return Z