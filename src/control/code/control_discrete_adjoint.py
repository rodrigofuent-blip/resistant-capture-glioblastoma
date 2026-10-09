"""Exact discrete RK4 adjoint and projected stationarity for interval-constant controls.
Reverse differentiates all 4 stages of the same augmented RK4 state/cost integration.
"""
import numpy as np
from numba import njit
from control_dynamics import P0,IC,W,forward,objective,at_init

@njit
def stage_pull(v,z,u,p,w):
 s,d,r=z[:3];N=s+d+r;a=p[6]+p[7]*u;q=1-N/p[10]
 j11=p[0]*q-p[0]*s/p[10]-p[3]*u-a
 j12=-p[0]*s/p[10]+p[8];j13=-p[0]*s/p[10]
 j21=-p[1]*d/p[10]+a;j22=p[1]*q-p[1]*d/p[10]-p[4]*u-p[8]-p[9]
 j23=-p[1]*d/p[10]
 j31=-p[2]*r/p[10];j32=-p[2]*r/p[10]+p[9]
 j33=p[2]*q-p[2]*r/p[10]-p[5]*u
 costp=v[3]+2*v[7]*max(0.,N-w[8])
 pulled=np.zeros(8)
 pulled[0]=j11*v[0]+j21*v[1]+j31*v[2]+costp
 pulled[1]=j12*v[0]+j22*v[1]+j32*v[2]+costp
 pulled[2]=j13*v[0]+j23*v[1]+j33*v[2]+costp+v[4]
 ub=-(p[3]+p[7])*s*v[0]+(-p[4]*d+p[7]*s)*v[1]-p[5]*r*v[2]+v[5]+2*u*v[6]
 return pulled,ub

@njit
def f8_local(y,u,p,w):
 s,d,r=y[:3];N=s+d+r;a=p[6]+p[7]*u;q=1-N/p[10]
 e=max(0.,N-w[8]);v=np.zeros(8)
 v[0]=p[0]*s*q-p[3]*u*s-a*s+p[8]*d
 v[1]=p[1]*d*q-p[4]*u*d+a*s-(p[8]+p[9])*d
 v[2]=p[2]*r*q-p[5]*u*r+p[9]*d
 v[3]=N;v[4]=r;v[5]=u;v[6]=u*u;v[7]=e*e
 return v

@njit
def exact_discrete_gradient(y,u,h,p,w):
 m=len(u);gr=np.zeros(m);la=np.zeros(8)
 s,d,r=y[-1,:3];N=s+d+r+w[6];base=w[3]-w[5]*r/(N*N)
 la[0]=base;la[1]=base;la[2]=base+w[4]+w[5]/N
 la[3]=w[0];la[4]=w[1];la[5]=0.;la[6]=w[2];la[7]=w[7]
 for k in range(m-1,-1,-1):
  z1=y[k];uk=u[k];k1=f8_local(z1,uk,p,w)
  z2=z1+.5*h*k1;k2=f8_local(z2,uk,p,w)
  z3=z1+.5*h*k2;k3=f8_local(z3,uk,p,w)
  z4=z1+h*k3
  l1=(h/6.)*la;l2=(h/3.)*la;l3=(h/3.)*la;l4=(h/6.)*la
  old=la.copy();du=0.
  p4,v4=stage_pull(l4,z4,uk,p,w);du+=v4;old+=p4;l3+=h*p4
  p3,v3=stage_pull(l3,z3,uk,p,w);du+=v3;old+=p3;l2+=.5*h*p3
  p2,v2=stage_pull(l2,z2,uk,p,w);du+=v2;old+=p2;l1+=.5*h*p2
  p1,v1=stage_pull(l1,z1,uk,p,w);du+=v1;old+=p1
  la=old;gr[k]=du
 return gr

def solve_exact(scenario,init,dt,max_iter=1000,tol=1e-5,relax=.05,wsafe=0.,nsafe=.52,dr_ratio=0.,seed_control=None):
 T=300.;m=int(round(T/dt));assert abs(m*dt-T)<1e-9
 p=P0.copy();p[5]=p[3]*dr_ratio;w=W[scenario].copy();w[7]=wsafe;w[8]=nsafe
 if seed_control is not None:u=np.asarray(seed_control,float).copy()
 elif init=='AT':u=at_init(m,dt,p,IC)
 elif init=='half':u=np.full(m,.5)
 elif init=='MTD':u=np.ones(m)
 else:raise ValueError(init)
 assert u.shape==(m,)
 hist_J=[];hist_R=[];hist_step=[];bestJ=np.inf;bestit=0;bu=u.copy()
 for it in range(max_iter):
  y,_=forward(u,dt,p,w,IC);grad=exact_discrete_gradient(y,u,dt,p,w)
  proj=np.clip(u-grad/(2*dt*w[2]),0.,p[11]);res=float(np.max(np.abs(proj-u)))
  un=(1-relax)*u+relax*proj;change=float(np.max(np.abs(un-u)))
  yn,_=forward(un,dt,p,w,IC);j=float(objective(yn,w))
  hist_J.append(j);hist_R.append(res);hist_step.append(change)
  if j<bestJ:bestJ=j;bestit=it+1;bu=un.copy()
  u=un
  if res<tol:break
 yl,_=forward(u,dt,p,w,IC);yb,_=forward(bu,dt,p,w,IC)
 rlast=float(np.max(abs(np.clip(u-exact_discrete_gradient(yl,u,dt,p,w)/(2*dt*w[2]),0.,p[11])-u)))
 rbest=float(np.max(abs(np.clip(bu-exact_discrete_gradient(yb,bu,dt,p,w)/(2*dt*w[2]),0.,p[11])-bu)))
 return dict(scenario=scenario,init=init,dt=dt,wsafe=wsafe,nsafe=nsafe,dr_ratio=dr_ratio,best_J=bestJ,last_J=float(objective(yl,w)),
    iterations=len(hist_J),best_iteration=bestit,best_residual=rbest,last_residual=rlast,converged=rlast<tol,
    u=bu,y=yb,u_last=u,y_last=yl,hist_J=np.array(hist_J),hist_R=np.array(hist_R),hist_step=np.array(hist_step))
