"""Numerical implementation for piecewise-constant resistance-aware optimal control.
Python 3; NumPy, SciPy and Numba. This is NOT a global optimizer.
Control is stored only on N=T/dt intervals (u[k] active on [t[k],t[k+1])).
Forward RK4 state and cost accumulators share exact same u[k] at every stage.
Adjoint uses u[k] throughout each interval, and numerically sampled mid-interval state.
Projected Pontryagin midpoint update is an approximation to interval-wise stationarity.
'projected_residual' is tested without multiplying by relaxation factor.
"""
from __future__ import annotations
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from pathlib import Path
import csv, json, hashlib, time, sys

# p: rS,rD,rR,dS,dD,dR,alpha0,alpha1,beta,eta,K,umax
P0=np.array([1e-3,1e-4,1.2e-3,.4469,.022345,0.,1e-5,.356,.2,.01,1.,1.],float)
IC=np.array([.5,0.,.02],float)
# w: wN,wR,wu,wT,wRT,wphi,eps,wSafe,Nsafe
W={
 'A':np.array([.5,10,.2,1,20,20,1e-8,0.,.5]),
 'B':np.array([.3,15,.25,.5,30,40,1e-8,0.,.5]),
 'C':np.array([1.,8,.15,3.,15.,10.,1e-8,0.,.5])}

@njit
def f3(x,u,p):
 s,d,r=x[0],x[1],x[2]; n=s+d+r; a=p[6]+p[7]*u; q=1.-n/p[10]
 return np.array([p[0]*s*q-p[3]*u*s-a*s+p[8]*d,
                  p[1]*d*q-p[4]*u*d+a*s-(p[8]+p[9])*d,
                  p[2]*r*q-p[5]*u*r+p[9]*d])

@njit
def f8(y,u,p,w):
 v=np.zeros(8);v[:3]=f3(y[:3],u,p)
 n=y[0]+y[1]+y[2]; excess=max(0.,n-w[8]);
 v[3]=n;v[4]=y[2];v[5]=u;v[6]=u*u;v[7]=excess*excess
 return v

@njit
def step8(y,u,h,p,w):
 k1=f8(y,u,p,w);k2=f8(y+.5*h*k1,u,p,w);k3=f8(y+.5*h*k2,u,p,w);k4=f8(y+h*k3,u,p,w)
 return y+(h/6.)*(k1+2*k2+2*k3+k4)

@njit
def forward(u,h,p,w,ic):
 m=len(u);y=np.zeros((m+1,8)); mid=np.zeros((m,3));y[0,:3]=ic
 for k in range(m):
  mid[k]=step8(y[k],u[k],.5*h,p,w)[:3]
  y[k+1]=step8(y[k],u[k],h,p,w)
 return y,mid

@njit
def terminal_grad(x,w):
 s,d,r=x; total=s+d+r+w[6]; base=w[3]-w[5]*r/(total*total)
 return np.array([base,base,base+w[4]+w[5]/total])

@njit
def adjrhs(lam,x,u,p,w):
 s,d,r=x; N=s+d+r; ls,ld,lr=lam
 a=p[6]+p[7]*u; nfac=1-N/p[10]; z=2*w[7]*max(0.,N-w[8]);
 return np.array([-(w[0]+z+ls*(p[0]*nfac-p[0]*s/p[10]-p[3]*u-a)+ld*(-p[1]*d/p[10]+a)+lr*(-p[2]*r/p[10])),
                  -(w[0]+z+ls*(-p[0]*s/p[10]+p[8])+ld*(p[1]*nfac-p[1]*d/p[10]-p[4]*u-p[8]-p[9])+lr*(-p[2]*r/p[10]+p[9])),
                  -(w[0]+w[1]+z+ls*(-p[0]*s/p[10])+ld*(-p[1]*d/p[10])+lr*(p[2]*nfac-p[2]*r/p[10]-p[5]*u))])

@njit
def adjoint_step(y_right,xright,xmid,xleft,u,h,p,w):
 # backwards half RK4: midpoint lambda with midpoint state approximated by endpoint average
 h2=-.5*h
 xq=.5*(xright+xmid)
 k1=adjrhs(y_right,xright,u,p,w)
 k2=adjrhs(y_right+.5*h2*k1,xq,u,p,w)
 k3=adjrhs(y_right+.5*h2*k2,xq,u,p,w)
 k4=adjrhs(y_right+h2*k3,xmid,u,p,w)
 lm=y_right+(h2/6)*(k1+2*k2+2*k3+k4)
 xq=.5*(xmid+xleft)
 k1=adjrhs(lm,xmid,u,p,w)
 k2=adjrhs(lm+.5*h2*k1,xq,u,p,w)
 k3=adjrhs(lm+.5*h2*k2,xq,u,p,w)
 k4=adjrhs(lm+h2*k3,xleft,u,p,w)
 ll=lm+(h2/6)*(k1+2*k2+2*k3+k4)
 return ll,lm

@njit
def backward(y,mid,u,h,p,w):
 m=len(u);lam=np.zeros((m+1,3));lmid=np.zeros((m,3));lam[-1]=terminal_grad(y[-1,:3],w)
 for k in range(m-1,-1,-1):
  lam[k],lmid[k]=adjoint_step(lam[k+1],y[k+1,:3],mid[k],y[k,:3],u[k],h,p,w)
 return lam,lmid

@njit
def projected(mid,lmid,p,w):
 m=len(mid);target=np.zeros(m)
 for k in range(m):
  s,d,r=mid[k];ls,ld,lr=lmid[k]
  z=(ls*(p[3]+p[7])*s+ld*(p[4]*d-p[7]*s)+lr*p[5]*r)/(2*w[2])
  target[k]=min(p[11],max(0.,z))
 return target

@njit
def objective(y,w):
 Y=y[-1];n=Y[0]+Y[1]+Y[2];r=Y[2]
 return w[0]*Y[3]+w[1]*Y[4]+w[2]*Y[6]+w[7]*Y[7]+w[3]*n+w[4]*r+w[5]*r/(n+w[6])

@njit
def at_init(m,h,p,ic):
 # Hysteretic AT initial guess for the control optimization.
 u=np.zeros(m);x=ic.copy();on=False
 for k in range(m):
  n=x[0]+x[1]+x[2]
  if on and n<=.45:on=False
  elif not on and n>=.55:on=True
  u[k]=p[11] if on else 0.
  # This initializer uses RK4 with interval-constant treatment.
  f1=f3(x,u[k],p);f2=f3(x+.5*h*f1,u[k],p)
  f3v=f3(x+.5*h*f2,u[k],p);f4=f3(x+h*f3v,u[k],p)
  x=x+h*(f1+2*f2+2*f3v+f4)/6
 return u

def solve(scenario,init,dt,max_iter=1000,relax=.05,tol=1e-5,wsafe=0.,nsafe=.52,dr_ratio=0.,seed_control=None,stagnant=0):
 T=300.; m=round(T/dt);assert abs(m*dt-T)<1e-10
 p=P0.copy();p[5]=p[3]*dr_ratio;w=W[scenario].copy();w[7]=wsafe;w[8]=nsafe
 if seed_control is not None: u=np.asarray(seed_control,float).copy();assert len(u)==m
 elif init=='AT':u=at_init(m,dt,p,IC)
 elif init=='half':u=np.full(m,.5*p[11]);
 elif init=='MTD':u=np.full(m,p[11]);
 else:raise ValueError(init)
 if np.any((u<0)|(u>p[11])):raise ValueError('inadmissible seed')
 bestJ=float('inf');bestit=-1;bestU=u.copy();he=[];hr=[];hj=[]
 for it in range(max_iter):
  y,mid=forward(u,dt,p,w,IC);lam,lmid=backward(y,mid,u,dt,p,w);unproj=projected(mid,lmid,p,w)
  residual=float(np.max(np.abs(unproj-u)))
  un=(1-relax)*u+relax*unproj;change=float(np.max(np.abs(un-u)))
  yn,_=forward(un,dt,p,w,IC);J=float(objective(yn,w))
  he.append(change);hr.append(residual);hj.append(J)
  if J<bestJ:bestJ=J;bestit=it+1;bestU=un.copy()
  u=un
  if residual<tol:break
 ylast,mlast=forward(u,dt,p,w,IC);ll,lm=backward(ylast,mlast,u,dt,p,w)
 lastResidual=float(np.max(np.abs(projected(mlast,lm,p,w)-u)))
 by,bmid=forward(bestU,dt,p,w,IC);bl,bm=backward(by,bmid,bestU,dt,p,w)
 bestResidual=float(np.max(np.abs(projected(bmid,bm,p,w)-bestU)))
 return {'scenario':scenario,'init':init,'dt':dt,'iterations':len(he),'best_iteration':bestit,
  'converged':lastResidual<tol,'last_residual':lastResidual,'best_residual':bestResidual,
  'last_change':he[-1],'best_J':bestJ,'last_J':float(objective(ylast,w)),
  'max_hamiltonian_update':max(hr),'T_final':m*dt,'u':bestU,'y':by,
  'hist_err':np.array(he),'hist_resid':np.array(hr),'hist_J':np.array(hj),
  'wsafe':wsafe,'nsafe':nsafe,'dr_ratio':dr_ratio,'w':w,'p':p,'u_last':u,'y_last':ylast}

def dop853_eval(u,dt,p,w):
 y=np.r_[IC,np.zeros(5)]; max_state=0.
 for k,uk in enumerate(u):
  # interval-wise independent integrator and objective quadrature
  def f(t,v):
   s,d,r=v[:3];n=s+d+r;a=p[6]+p[7]*uk;h=1.-n/p[10]
   return [p[0]*s*h-p[3]*uk*s-a*s+p[8]*d,
           p[1]*d*h-p[4]*uk*d+a*s-(p[8]+p[9])*d,
           p[2]*r*h-p[5]*uk*r+p[9]*d,
           n,r,uk,uk*uk,max(0.,n-w[8])**2]
  so=solve_ivp(f,(k*dt,(k+1)*dt),y,method='DOP853',rtol=1e-10,atol=1e-12)
  if not so.success:raise RuntimeError(so.message)
  y=so.y[:,-1]
 N=y[:3].sum();R=y[2];J=w[0]*y[3]+w[1]*y[4]+w[2]*y[6]+w[7]*y[7]+w[3]*N+w[4]*R+w[5]*R/(N+w[6])
 return J,y

def summarize(z):
 u=z['u'];y=z['y'];last=y[-1]; n=last[:3].sum();r=last[2];dt=z['dt'];
 above=u>=.5; rising=np.flatnonzero(above&np.r_[True,~above[:-1]]);
 return {k:z[k] for k in ['scenario','init','dt','iterations','best_iteration','converged','last_residual','best_residual','last_change','best_J','last_J','T_final','wsafe','nsafe','dr_ratio']}|{
  'phi_R_T':r/n,'N_T':n,'R_T':r,'A_N':last[3],'A_R':last[4],
  'A_u':last[5],'A_u2':last[6],'A_safe':last[7],
  'N_max_nodes':float(y[:,:3].sum(axis=1).max()),'n_on_segments':len(rising),
  'first_pulse_start':float(rising[0]*dt) if len(rising) else float('nan')}

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--max-iter',type=int,default=1000);args=ap.parse_args();out=Path(args.out).resolve()
 protected=Path(__file__).resolve().parents[1].resolve()
 if protected==out or protected in out.parents:
  ap.error('--out must not point into the packaged numerical reference directory')
 out.mkdir(parents=True,exist_ok=True)
 print('versions',np.__version__, flush=True)
 runs=[]
 # Baseline 9 runs; refinement selected 6 runs; safety three + one refinement.
 plan=[(s,ini,.5,0.) for s in 'ABC' for ini in ('AT','half','MTD')]
 plan += [(s,ini,.1,0.) for s,ini in [('A','AT'),('A','half'),('B','AT'),('B','half'),('C','AT'),('C','MTD')]]
 plan += [('B',ini,.5,1000.) for ini in ('AT','half','MTD')]
 plan += [('B','half',.25,1000.)]
 for s,ini,dt,wsafe in plan:
  start=time.time();z=solve(s,ini,dt,max_iter=args.max_iter,wsafe=wsafe)
  key=f'{s}_{ini}_dt{dt:g}_safe{wsafe:g}';np.savez_compressed(out/f'{key}.npz',u=z['u'],y=z['y'],u_last=z['u_last'],y_last=z['y_last'],hist_err=z['hist_err'],hist_resid=z['hist_resid'],hist_J=z['hist_J'])
  row=summarize(z);row['duration_s']=time.time()-start;runs.append(row);print(key,{'bestJ':row['best_J'],'last_res':row['last_residual'],'conv':row['converged'],'niter':row['iterations']},flush=True)
 with open(out/'unified_summary.csv','w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(runs[0]));wr.writeheader();wr.writerows(runs)
