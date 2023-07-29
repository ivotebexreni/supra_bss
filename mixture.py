import numpy as np
import spec
import thevenin as th

pi2=2*np.pi

def mixmatrix_1(Zr,Zd):
  
  # Impedâncias da rede
  # Zr = (a,b,c,np.array((d,e)))
  # a: Impedância primária da fonte
  # b: Impedância secundária da fonte
  # c: Impedância do barramento ao terra (será considerada infinita)
  # d: Impedância da linha 1
  # e: Impedância da linha 2

  # Impedâncias dos equipamentos
  # Zd = ((np.array((a,b)),np.array((c,d))),np.array((e,f)))
  # a: Impedância primária do equipamento 1
  # b: Impedância primária do equipamento 2
  # c: Impedância secundária do equipamento 1
  # d: Impedância secundária do equipamento 2
  # e: Barra de ligação do equipamento 1
  # f: Barra de ligação do equipamento 2

  nr=np.array(Zr[3]).shape[0] # Número de linhas da rede
  nd=np.array(Zd[1]).shape[0] # Número de equipamentos, (não conta a alimentação da rede)

  Y=np.zeros((nd+nr+2,nd+nr+2),dtype='complex')

  Y[0,0]+=(1/Zr[0]+1/Zr[1])
  Y[1,1]+=(np.sum(1/Zr[3])+1/Zr[1]+1/Zr[2])

  for i in range(nr):
    Y[i+2,i+2]+=(1/Zr[3][i])

  for i in range(nd):
    Y[Zd[1][i],Zd[1][i]]+=(1/Zd[0][1][i])
    Y[nr+2+i,nr+2+i]+=(1/Zd[0][1][i])
    Y[nr+2+i,nr+2+i]+=(1/Zd[0][0][i])

  Y[0,1]-=(1/Zr[1])
  Y[1,0]-=(1/Zr[1])

  for i in range(nr):
    Y[1,i+2]-=(1/Zr[3][i])
    Y[i+2,1]-=(1/Zr[3][i])

  for i in range(nd):
    Y[Zd[1][i],nr+2+i]-=(1/Zd[0][1][i])
    Y[nr+2+i,Zd[1][i]]-=(1/Zd[0][1][i])

  # Dicionário que seleciona no vetor das correntes aquela que são fontes de harmônicos (alimentação da rede e equipamentos)
  E=np.zeros((nr+nd+2,nd+1))
  E[0,0]+=1
  E[2+nr:,1:]+=np.eye(nd)

  D0=np.zeros((nd+1,nr+nd+2))
  D0[0,0]=1
  D0[1:,2+nr:]=np.eye(nd)
  D1=np.zeros((nd+1,nr+nd+2))
  D1[0,1]=1
  for i in range(nd):
    D1[i+1,Zd[1][i]]=1
  D=D0-D1

  Z=np.concatenate(([Zr[1]],Zd[0][1]))

  # return np.linalg.inv(np.diag(Z)).dot(D).dot(np.linalg.inv(Y)).dot(E)
  return D.dot(np.linalg.inv(Y)).dot(E)

  # Z=np.linalg.inv(Y)
  # Y_=np.tile(1/np.concatenate(([Zr[1]],Zd[0][1])),(nd+1,1)).T
  # A=D.dot(Z.dot(E))*Y_
  
  # return A

yr=1/2.15
ks=1
kp=1
lim01=2.8
lim02=3.8
pi=np.pi

def resistance(R0,f):
  Xs=np.sqrt(8*pi*f*1e-7*ks/R0)
  Xp=np.sqrt(8*pi*f*1e-7*kp/R0)
  if Xs<=lim01:
    Ys=Xs**4/(192+.8*Xs**4)
  elif Xs>lim02:
    Ys=.354*Xs-.733
  else:
    Ys=-.136-.0177*Xs+.0563*Xs**2
  Yp=Xp**4/(192+.8*Xp**4)*yr*yr*2.9
  return R0*(1+(Ys+Yp)*1.5)
  # return R0

def impedance(s,f):
  # s='2.1-c-4.5'
  i=s.find('c')
  c='c'
  c = c if i>-1 else 'l'
  i = i if i>-1 else s.find('l')
  ss=s.split(c)
  Z=resistance(float(ss[0]),np.abs(f))
  # print(ss)
  X=2j*pi*f*float(ss[1])
  if c=='l':
    Z+=X
  elif c=='c':
    Z+=(1/X)
  return Z

# Zr = (a,b,c,np.array(d,e))
def transfXnet(p,s,line,g,f):
  Zp=impedance(p,f)
  Zs=impedance(s,f)
  Zg=impedance(g,f)
  Zl=[]
  for i in range(len(line)):
    Zl.append(impedance(line[i],f))
  Zl=np.array(Zl)
  Zr=(Zp,Zs,Zg,Zl)
  return Zr

# Zd = ((np.array((a,b)),np.array((c,d))),np.array((e,f)))
def transfXeq(eqp,eqs,f,n):
  n=np.array(n)
  p=[]
  s=[]
  for i in range(len(eqp)):
    p.append(impedance(eqp[i],f))
    s.append(impedance(eqs[i],f))
  p=np.array(p)
  s=np.array(s)
  Zd=((p,s),n)
  return Zd

def transfXeqfun(feqp,eqs,f,n):
  n=np.array(n)
  p=[]
  s=[]
  for i in range(len(eqs)):
    p.append(feqp[i]['real'](np.abs(f))+1j*(np.sign(f)*feqp[i]['imag'](np.abs(f))))
    s.append(impedance(eqs[i],f))
  p=np.array(p)
  s=np.array(s)
  Zd=((p,s),n)
  return Zd

def mixtensor(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n,f=np.arange(2000,150001,1000)):
  p,s,line,eqp,eqs=(Zp0,Zs0,Zl,Zpn,Zsn)
  N=len(eqp)+1
  A=np.zeros((f.shape[0],N,N),dtype='complex')
  for i in range(f.shape[0]):
    # A[i,:,:]=mixmatrix_1(transfXnet(p,s,line,Zg,f[i]),transfXeq(eqp,eqs,f[i],n))
    # A[i,:,:]=mixmatrix(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n,f[i])
    A[i,:,:]=mixmatrix_1(transfXnet(p,s,line,Zg,f[i]),transfXeqfun(eqp,eqs,f[i],n))
  return A

def mixmatrix(Zp0,Zs0,Zg,Zl,Zpn,Zsn,n,f):
  p,s,line,eqp,eqs=(Zp0,Zs0,Zl,Zpn,Zsn)
  return mixmatrix_1(transfXnet(p,s,line,Zg,f),transfXeq(eqp,eqs,f,n))

def mixsignals(s,mxtensor,fs=500e3,nperseg=500,noverlap=250,window='hann',return_onesided=True):
  f,t,S=spec.stft(s,fs=500e3,nperseg=nperseg,noverlap=noverlap,window=window,return_onesided=return_onesided)
  nf=mxtensor.shape[0]
  X=np.zeros(S.shape,dtype='complex')
  for i in range(nf):
    X[:,i,:]=mxtensor[i,:,:].dot(S[:,i,:])
  return f,t,S,X

def mixtensor1(fs,ns,D,d,de,n,p,b0,b1,b2,l0,Rg,lg,FP,Z0,Peq,FPeq,nl,corfat,gamma,f0=50,V=230):
  freq=np.fft.fftfreq(ns,d=1/fs)[1:]
  
  R0=p*D/(b0*1e-6) # Resistência do alimentador ao barramento central (Ohm)
  Re=p*d/(b1*1e-6) # Resistência dos condutores (Ohm)
  R2=p*de/(b2*1e-6) # Resistência dos secundários (Ohm)
  Rp0=Z0*FP # Resistência da rede
  Rep=V*V/Peq # Resistência dos equipamentos

  print(R0,Re,R2,Rp0,Rep)

  L0=l0*D*1e-3 # Indutância do alimentador ao barramento central (H)
  Le=l0*d*1e-3 # Indutância dos condutores (H)
  L2=l0*de*1e-3 # Indutância dos secundários (H)
  C0=1/(np.sqrt(Z0*Z0-Rp0*Rp0)*pi2*f0) # Capacitância da rede
  Ceq=1/(f0*pi2*np.sqrt(1-FPeq*FPeq)*Rep) # Capacitância dos equipamentos

  print(L0,Le,L2,C0,Ceq)

  mxtensor_=np.zeros((freq.shape[0],1+len(n),1+len(n)),dtype='complex') # Mixture tensors

  Zl=np.zeros(Re.shape,dtype='complex') # Impedância dos Condutores
  Zsn=np.zeros(R2.shape,dtype='complex') # Impedância secundária dos equipamentos
  Zpn_=np.zeros(Rep.shape,dtype='complex') # Impedância primária dos equipamentos
  Zp=np.zeros((freq.size,),dtype='complex') # Impedâncias primárias da rede
  Zthg=np.zeros((freq.size),dtype='complex') # Impedâncias de thevenin da rede
  Zpn=np.zeros((len(n),freq.size),dtype='complex') # Impedâncias de thevenin da rede
  Zthe=np.zeros((len(n),freq.size),dtype='complex') # Impedâncias de thevenin da rede

  for i,f in enumerate(freq):
    if i%100000==0: print(i)
    Zp0=resistance(Rp0,np.abs(f))+1/(1j*C0*pi2*f)
    Zs0=resistance(R0,np.abs(f))+1j*L0*pi2*f
    Zg=resistance(Rg,np.abs(f))+1j*lg*pi2*f
    for j in range(Zl.shape[0]):
      Zl[j]=resistance(Re[j],np.abs(f))+1j*Le[j]*pi2*f
    for j in range(Zsn.shape[0]):
      Zsn[j]=resistance(R2[j],np.abs(f))+1j*L2[j]*pi2*f
      Zpn_[j]=(resistance(Rep[j],np.abs(f))+1/(1j*Ceq[j]*pi2*f))*corfat[j]
      if nl[j]>1: Zpn_[j]/=nl[j]
    Zpn[:,i]=Zpn_
    Zp[i]=Zp0
    Zthg[i]=th.thevenineq_grid(Zp0,Zs0,Zg,Zl,Zpn_,Zsn,n)
    for j in range(Zsn.shape[0]):
      Zthe[j,i]=th.thevenineq_equip(Zp0,Zs0,Zg,Zl,Zpn_,Zsn,n,j)
    # Zth.append(th.thevenineq_equip(Zp0[i],Zs0,Zg,Zl,Zpn,Zsn,n,3))
    
    mxtensor_[i]=mixmatrix_1((Zp0,Zs0,Zg,Zl),((Zpn_,Zsn),n))
  # return mxtensor_,np.array(Zp0),np.array(Zth) # Para o caso de apenas a alimentação de linha ser um sinal de tensão, os qeuipamentos são sinais de corrente
  return mxtensor_,Zp,Zpn,Zthg,Zthe,C0,Ceq #,np.array(Zp0),np.array(Zth) # Para o caso de todas as fontes serem sinais de tensão

def impedance_tensor(fs,ns,D,d,de,n,p,b0,b1,b2,l0,Rg,lg,FP,Z0,Peq,FPeq,nl,corfat,gamma,f0=50,V=230):
  freq=np.fft.fftfreq(ns,d=1/fs)[1:]
  
  R0=p*D/(b0*1e-6) # Resistência do alimentador ao barramento central (Ohm)
  Re=p*d/(b1*1e-6) # Resistência dos condutores (Ohm)
  R2=p*de/(b2*1e-6) # Resistência dos secundários (Ohm)
  Rp0=Z0*FP # Resistência da rede
  Rep=V*V/Peq # Resistência dos equipamentos

  print(R0,Re,R2,Rp0,Rep)

  L0=l0*D*1e-3 # Indutância do alimentador ao barramento central (H)
  Le=l0*d*1e-3 # Indutância dos condutores (H)
  L2=l0*de*1e-3 # Indutância dos secundários (H)
  C0=1/(np.sqrt(Z0*Z0-Rp0*Rp0)*pi2*f0) # Capacitância da rede
  Ceq=1/(f0*pi2*np.sqrt(1-FPeq*FPeq)*Rep) # Capacitância dos equipamentos

  print(L0,Le,L2,C0,Ceq)

  mxtensor_=np.zeros((freq.shape[0],1+len(n),1+len(n)),dtype='complex') # Mixture tensors

  Zl=np.zeros(Re.shape,dtype='complex') # Impedância dos Condutores
  Zsn=np.zeros(R2.shape,dtype='complex') # Impedância secundária dos equipamentos
  Zpn_=np.zeros(Rep.shape,dtype='complex') # Impedância primária dos equipamentos
  Zp=np.zeros((freq.size,),dtype='complex') # Impedâncias primárias da rede
  Zthg=np.zeros((freq.size),dtype='complex') # Impedâncias de thevenin da rede
  Zpn=np.zeros((len(n),freq.size),dtype='complex') # Impedâncias de thevenin da rede
  Zthe=np.zeros((len(n),freq.size),dtype='complex') # Impedâncias de thevenin da rede

  for i,f in enumerate(freq):
    if i%100000==0: print(i)
    Zp0=resistance(Rp0,np.abs(f))+1/(1j*C0*pi2*f)
    Zs0=resistance(R0,np.abs(f))+1j*L0*pi2*f
    Zg=resistance(Rg,np.abs(f))+1j*lg*pi2*f
    for j in range(Zl.shape[0]):
      Zl[j]=resistance(Re[j],np.abs(f))+1j*Le[j]*pi2*f
    for j in range(Zsn.shape[0]):
      Zsn[j]=resistance(R2[j],np.abs(f))+1j*L2[j]*pi2*f
      Zpn_[j]=(resistance(Rep[j],np.abs(f))+1/(1j*Ceq[j]*pi2*f))*corfat[j]
      if nl[j]>1: Zpn_[j]/=nl[j]
    Zpn[:,i]=Zpn_
    Zp[i]=Zp0
    Zthg[i]=th.thevenineq_grid(Zp0,Zs0,Zg,Zl,Zpn_,Zsn,n)
    for j in range(Zsn.shape[0]):
      Zthe[j,i]=th.thevenineq_equip(Zp0,Zs0,Zg,Zl,Zpn_,Zsn,n,j)
    # Zth.append(th.thevenineq_equip(Zp0[i],Zs0,Zg,Zl,Zpn,Zsn,n,3))
    
    impedance_tens=np.zeros((ns-1,de.size+d.size+2,de.size+d.size+2),dtype='complex')
    for i in range(1,ns):
      impedance_tens[i-1]=impedance_matrix((Zp0,Zs0,Zg,Zl),((Zpn_,Zsn),n))
      
    return impedance_tens

    # mxtensor_[i]=mixmatrix_1((Zp0,Zs0,Zg,Zl),((Zpn_,Zsn),n))
  

def impedance_matrix(Zr,Zd):

  nr=np.array(Zr[3]).shape[0] # Número de linhas da rede
  nd=np.array(Zd[1]).shape[0] # Número de equipamentos, (não conta a alimentação da rede)

  Y=np.zeros((nd+nr+2,nd+nr+2),dtype='complex')

  Y[0,0]+=(1/Zr[0]+1/Zr[1])
  Y[1,1]+=(np.sum(1/Zr[3])+1/Zr[1]+1/Zr[2])

  for i in range(nr):
    Y[i+2,i+2]+=(1/Zr[3][i])

  for i in range(nd):
    Y[Zd[1][i],Zd[1][i]]+=(1/Zd[0][1][i])
    Y[nr+2+i,nr+2+i]+=(1/Zd[0][1][i])
    Y[nr+2+i,nr+2+i]+=(1/Zd[0][0][i])

  Y[0,1]-=(1/Zr[1])
  Y[1,0]-=(1/Zr[1])

  for i in range(nr):
    Y[1,i+2]-=(1/Zr[3][i])
    Y[i+2,1]-=(1/Zr[3][i])

  for i in range(nd):
    Y[Zd[1][i],nr+2+i]-=(1/Zd[0][1][i])
    Y[nr+2+i,Zd[1][i]]-=(1/Zd[0][1][i])

  return np.linalg.inv(Y)