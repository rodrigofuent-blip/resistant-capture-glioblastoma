"""Independent PyTorch reverse-mode differentiation of the EXACT RK4-discretized objective.
This does not use FBS continuous-adjoint equations or midpoint projection.
"""
import importlib.util,numpy as np,torch,csv
from pathlib import Path
b=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p1r4b',b/'code/control_dynamics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(1)
def exact_gradient(u_np,dt,p,w):
 u=torch.tensor(u_np,dtype=torch.float64,requires_grad=True)
 y=torch.tensor([.5,0.,.02,0.,0.,0.,0.,0.],dtype=torch.float64)
 p=torch.tensor(p,dtype=torch.float64);w=torch.tensor(w,dtype=torch.float64)
 def ff(y,uk):
  s,d,r=y[0],y[1],y[2];n=s+d+r;a=p[6]+p[7]*uk;crowd=1-n/p[10]
  fs=p[0]*s*crowd-p[3]*uk*s-a*s+p[8]*d
  fd=p[1]*d*crowd-p[4]*uk*d+a*s-(p[8]+p[9])*d
  fr=p[2]*r*crowd-p[5]*uk*r+p[9]*d
  e=torch.relu(n-w[8]);return torch.stack((fs,fd,fr,n,r,uk,uk*uk,e*e))
 for uk in u:
  k1=ff(y,uk);k2=ff(y+.5*dt*k1,uk);k3=ff(y+.5*dt*k2,uk);k4=ff(y+dt*k3,uk)
  y=y+(dt/6)*(k1+2*k2+2*k3+k4)
 n=y[:3].sum();r=y[2]
 J=w[0]*y[3]+w[1]*y[4]+w[2]*y[6]+w[7]*y[7]+w[3]*n+w[4]*r+w[5]*r/(n+w[6])
 J.backward()
 return float(J.detach()),u.grad.detach().numpy().copy()
out=Path(__file__).resolve().parents[3]/'work/standalone_autograd'
out.mkdir(parents=True,exist_ok=True)
rows=[]
for name in ['A_AT_dt0.5_safe0','A_half_dt0.5_safe0','B_AT_dt0.5_safe0','C_MTD_dt0.5_safe0','B_half_dt0.5_safe1000']:
 z=np.load(b/'results'/f'{name}.npz');s=name[0];dt=.5;w=m.W[s].copy();w[7]=1000 if 'safe1000' in name else 0.;w[8]=.52
 for label in ['u','u_last']:
  u=z[label];J,g=exact_gradient(u,dt,m.P0,w);scaled=g/dt
  target=np.clip(u-g/(2*dt*w[2]),0.,1.)
  resid=float(np.max(abs(target-u)))
  y,mid=m.forward(u,dt,m.P0,w,m.IC);lam,lmid=m.backward(y,mid,u,dt,m.P0,w)
  approx=m.projected(mid,lmid,m.P0,w)
  row=dict(case=name,iterate=label,J_exact_autograd=J,J_numba=float(m.objective(y,w)),J_diff=float(abs(J-m.objective(y,w))),discrete_projected_residual=resid,midpoint_projected_residual=float(max(abs(approx-u))),max_abs_discrete_vs_midpoint_grad=float(np.max(np.abs(scaled-(2*w[2]*u-(lmid[:,0]*(m.P0[3]+m.P0[7])*mid[:,0]+lmid[:,1]*(m.P0[4]*mid[:,1]-m.P0[7]*mid[:,0])+lmid[:,2]*m.P0[5]*mid[:,2]))))),u_exposure=float(y[-1,5]))
  rows.append(row);print(row,flush=True)
with open(out/'exact_discrete_autograd.csv','w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
