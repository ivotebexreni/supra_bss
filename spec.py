import numpy as np
from scipy import signal as sp
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cmap=LinearSegmentedColormap.from_list('hsvcmap',[(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0),(.5,0,1)],N=100)

def stft(x,fs,nperseg=None,noverlap=None,window='hann',
         return_onesided=True,boundary='zeros',axis=-1):
  nperseg = nperseg if nperseg is not None else int(np.ceil(x.shape[0]/10))
  noverlap = noverlap if noverlap is not None else nperseg//2
  return sp.stft(x,fs=fs,window=window,nperseg=nperseg,noverlap=noverlap,
                 nfft=None,detrend=False,return_onesided=return_onesided,
                 boundary=boundary,padded=True,axis=axis)

def istft(X,fs,nperseg=None,noverlap=None,window='hann',
         input_onesided=True,boundary='zeros',axis=-1):
  nperseg = nperseg if nperseg is not None else int(np.ceil(X.shape[0]/10))
  noverlap = noverlap if noverlap is not None else nperseg//2
  return sp.istft(X,fs=fs,window=window,nperseg=nperseg,noverlap=noverlap,
                 nfft=None,input_onesided=input_onesided,
                 boundary=boundary)

def spectrogram(x,fs,nperseg=None,noverlap=None,window='hann',
         return_onesided=True,boundary=None,axis=-1,density=True,dB=True):
  f,t,S=stft(x,fs,nperseg=nperseg,noverlap=noverlap,
             window=window,return_onesided=return_onesided,boundary=boundary,axis=axis)
  S=np.abs(S)
  S = S*S if density else S
  S = 10*np.log10(S) if dB else S
  return f,t,S
  
def plotspec(x,fs=500e3,flim=(2e3,150e3),nperseg=500,noverlap=450,
             window='hann',density=True,cmap=cmap,vlim=(-180,-40),label=None,colorbar=True,dB=True,rasterized=True,
             timeunit='ms',frequnit='kHz'):
  nperseg = nperseg if nperseg is not None else int(np.ceil(x.shape[0]/10))
  noverlap = noverlap if noverlap is not None else int(nperseg/2)
  f,t,S=spectrogram(x,fs,nperseg=nperseg,noverlap=noverlap,window=window,
                    return_onesided=True,boundary=None,axis=-1,
                    density=density,dB=dB)
  tscale = 1 if timeunit=='s' else 1000
  fscale = 1 if frequnit=='Hz' else 1/1000
  flim=flim if flim is not None else (np.min(f),np.max(f))
  fr=fs/nperseg
  fi=[int(np.floor(flim[0]/fr)),int(np.ceil(flim[1]/fr))]
  fi[1]=fi[1] if fi[1]<(nperseg//2) else fi[1]-1
  S=S[fi[0]:fi[1],:]
  f=f[fi[0]:fi[1]]
  f=np.r_[f-(f[1]-f[0])/2,f[-1]+(f[1]-f[0])/2]
  t=np.r_[0,t+t[0]]
  vlim=(S.min(),S.max()) if vlim is None else vlim
  plt.pcolormesh(t*tscale,f*fscale,S,cmap=cmap,vmin=vlim[0],vmax=vlim[1],rasterized=rasterized)
  # plt.pcolormesh(t,f,S,cmap=cmap)
  if colorbar:
    cbar=plt.colorbar()
    if label is not None:
      cbar.set_label(label)
  plt.xlabel('tempo ('+timeunit+')')
  plt.ylabel('Frequência ('+frequnit+')')
  # return f,t,S

def plotspecfreq(f_,t_,S,flim=(0,1),cmap=cmap,vlim=(-180,-40),label=None,
                 mag=True,colorbar=True,density=True,dB=True,rasterized=True,
                 timeunit='ms',frequnit='kHz'):
  Ps=S.copy()
  f=f_.copy()
  t=t_.copy()
  tscale = 1000 if timeunit=='ms' else 1
  fscale = 1/1000 if frequnit=='kHz' else 1
  if density: Ps*=Ps
  if mag: Ps=np.abs(Ps)
  if dB: Ps=10*np.log10(Ps)
  f=np.r_[f-(f[1]-f[0])/2,f[-1]+(f[1]-f[0])/2]*fscale
  t=np.r_[0,t+t[0]]*tscale
  # flim=flim if flim is not None else (np.min(f),np.max(f))
  vlim=(Ps.min(),Ps.max()) if vlim is None else vlim
  plot=plt.pcolormesh(t,f[int(flim[0]*f.shape[0]):int(flim[1]*f.shape[0])],
                 Ps[int(flim[0]*f.shape[0]):int(flim[1]*f.shape[0]),:],
                 cmap=cmap,vmin=vlim[0],vmax=vlim[1],rasterized=rasterized)
  if colorbar:
    cbar=plt.colorbar()
    if label is not None:
      cbar.set_label(label)
  plt.xlabel('tempo ('+timeunit+')')
  plt.ylabel('Frequência ('+frequnit+')')
  # return plot

def plotfft(x,fs=500e3,dB=True,density=False,onesided=True,**kwargs):
  X=np.fft.fft(x)
  n=X.shape[0]
  f=np.fft.fftfreq(n,1/fs)
  f[f.shape[0]//2]*=-1
  if onesided:
    f=f[:f.shape[0]//2+1]
    X=X[:X.shape[0]//2+1]
  X=np.abs(X)
  if density: X*=X
  if dB: X=10*np.log10(X)
  plt.plot(f,X,**kwargs)
  # return f