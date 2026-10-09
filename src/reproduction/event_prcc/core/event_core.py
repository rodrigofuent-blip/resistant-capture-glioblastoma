import numpy as np
from numba import njit
BASE=np.array([1e-3,1e-4,1.2e-3,0.4469,0.05*0.4469,0.0,1e-5,0.356,0.2,0.01,1.0,0.55,0.45],float)
COLS=['phi_R_T','A_R','A_N','A_u','N_T','n_activations','n_deactivations','first_on','t_final','S_T','D_T','R_T','final_on']
@njit(cache=True,inline='always')
def deriv(S,D,R,u,p):
 N=S+D+R;a=p[6]+p[7]*u
 return (p[0]*S*(1-N/p[10])-p[3]*u*S-a*S+p[8]*D,p[1]*D*(1-N/p[10])-p[4]*u*D+a*S-(p[8]+p[9])*D,p[2]*R*(1-N/p[10])-p[5]*u*R+p[9]*D,R,N,u)
@njit(cache=True,inline='always')
def rk(S,D,R,AR,AN,AU,u,h,p):
 k1=deriv(S,D,R,u,p);k2=deriv(S+.5*h*k1[0],D+.5*h*k1[1],R+.5*h*k1[2],u,p);k3=deriv(S+.5*h*k2[0],D+.5*h*k2[1],R+.5*h*k2[2],u,p);k4=deriv(S+h*k3[0],D+h*k3[1],R+h*k3[2],u,p)
 S2=S+h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6;D2=D+h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6;R2=R+h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6;AR2=AR+h*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6;AN2=AN+h*(k1[4]+2*k2[4]+2*k3[4]+k4[4])/6;AU2=AU+h*(k1[5]+2*k2[5]+2*k3[5]+k4[5])/6
 if S2<0:S2=0.
 if D2<0:D2=0.
 if R2<0:R2=0.
 return S2,D2,R2,AR2,AN2,AU2
@njit(cache=True,inline='always')
def itau(S,D,R,AR,AN,AU,u,tau,p):return rk(S,D,R,AR,AN,AU,u,tau,p)
@njit(cache=True)
def event_sim(p,S0,D0,R0,T,hmax=.1,tol=1e-11):
 S=S0;D=D0;R=R0;AR=0.;AN=0.;AU=0.;t=0.;on=(S+D+R)>=p[11];n_on=0;n_off=0;first=np.nan
 for _ in range(int(np.ceil(T/hmax))+20):
  if t>=T-1e-13:break
  rem=min(hmax,T-t)
  for _inner in range(8):
   if rem<=1e-14:break
   u=1. if on else 0.;thr=p[12] if on else p[11];direction=-1. if on else 1.;g0=((S+D+R)-thr)*direction;e=itau(S,D,R,AR,AN,AU,u,rem,p);g1=((e[0]+e[1]+e[2])-thr)*direction
   if g1<0:S,D,R,AR,AN,AU=e;t+=rem;rem=0.;break
   if g0>=0:tau=0.;ee=(S,D,R,AR,AN,AU)
   else:
    lo=0.;hi=rem
    for _b in range(60):
     mid=.5*(lo+hi);m=itau(S,D,R,AR,AN,AU,u,mid,p);gm=((m[0]+m[1]+m[2])-thr)*direction
     if gm>=0:hi=mid
     else:lo=mid
     if hi-lo<=tol:break
    tau=.5*(lo+hi);ee=itau(S,D,R,AR,AN,AU,u,tau,p)
   S,D,R,AR,AN,AU=ee;t+=tau;rem-=tau;on=not on
   if on:
    n_on+=1
    if np.isnan(first):first=t
   else:n_off+=1
   if tau<=1e-14 and rem>1e-14:
    eps=min(rem,1e-10);S,D,R,AR,AN,AU=itau(S,D,R,AR,AN,AU,1. if on else 0.,eps,p);t+=eps;rem-=eps
  else:return np.full(13,np.nan)
 N=S+D+R;return np.array([R/N,AR,AN,AU,N,float(n_on),float(n_off),first,t,S,D,R,1. if on else 0.])
@njit(cache=True)
def archived_node_sim(p,S0,D0,R0,T,dt):
 n=int(np.ceil(T/dt))+1;S=S0;D=D0;R=R0;ARa=0.;ANa=0.;AUa=0.;on=False;n_on=0;n_off=0;first=np.nan;sumR=0.;sumN=0.;sumU=0.;Rf=Nf=Uf=Rl=Nl=Ul=0.
 for k in range(n):
  N=S+D+R;old=on;on=(N>p[12]) if on else (N>=p[11])
  if k>0:
   if on and not old:
    n_on+=1
    if np.isnan(first):first=k*dt
   if old and not on:n_off+=1
  u=1. if on else 0.;sumR+=R;sumN+=N;sumU+=u
  if k==0:Rf=R;Nf=N;Uf=u
  if k==n-1:Rl=R;Nl=N;Ul=u
  if k<n-1:S,D,R,ARa,ANa,AUa=rk(S,D,R,ARa,ANa,AUa,u,dt,p)
 AR=dt*(sumR-.5*(Rf+Rl));AN=dt*(sumN-.5*(Nf+Nl));AU=dt*(sumU-.5*(Uf+Ul));N=S+D+R
 return np.array([R/N,AR,AN,AU,N,float(n_on),float(n_off),first,(n-1)*dt,S,D,R,1. if on else 0.])
@njit(cache=True)
def mtd_sim(p,S0,D0,R0,T,h=.1):
 S=S0;D=D0;R=R0;AR=0.;AN=0.;AU=0.;t=0.
 while t<T-1e-13:
  hh=min(h,T-t);S,D,R,AR,AN,AU=rk(S,D,R,AR,AN,AU,1.,hh,p);t+=hh
 N=S+D+R;return np.array([R/N,AR,AN,AU,N,t,S,D,R])
# compile signatures lazily/cache
