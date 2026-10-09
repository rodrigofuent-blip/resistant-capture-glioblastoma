import importlib.util,numpy as np,glob,json,csv
from pathlib import Path
b=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('p1r4b',b/'code/control_dynamics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path(__file__).resolve().parents[3]/'work/standalone_baseline_checks'
out.mkdir(parents=True,exist_ok=True)
rows=[];grad_rows=[];dop_rows=[]
for fn in sorted((b/'results').glob('*_dt0.5_safe0.npz')):
 s,ini=fn.name.split('_')[0:2];z=np.load(fn);u=z['u'];w=m.W[s];y,mid=m.forward(u,.5,m.P0,w,m.IC);la,lmid=m.backward(y,mid,u,.5,m.P0,w)
 p=m.projected(mid,lmid,m.P0,w);res=float(np.max(abs(p-u)))
 J=float(m.objective(y,w));assert abs(J-z['hist_J'].min())<1e-9
 row={'case':fn.stem,'best_J':J,'last_J_history':float(z['hist_J'][-1]),'best_residual':res,'last_err':float(z['hist_err'][-1]),'last_resid_history':float(z['hist_resid'][-1]),'iterations':len(z['hist_J']),'A_u':float(y[-1,5]),'A_u2':float(y[-1,6]),'A_N':float(y[-1,3]),'A_R':float(y[-1,4]),'phi_R':float(y[-1,2]/sum(y[-1,:3])),'max_N':float(y[:,:3].sum(axis=1).max())};rows.append(row)
 print('CASE',row,flush=True)
 # Selected controls varied at 2 or 3 intervals; finite-difference gradient for interior controls if present
 selected=[0,12,30,160,250,400]
 for k in selected:
  if k>=len(u):continue
  h=1e-6;lo=max(0.,u[k]-h);hi=min(1.,u[k]+h)
  if hi-lo<1e-14:continue
  uu=u.copy();uu[k]=hi; yp,_=m.forward(uu,.5,m.P0,w,m.IC);jp=m.objective(yp,w)
  uu[k]=lo; ym,_=m.forward(uu,.5,m.P0,w,m.IC);jm=m.objective(ym,w)
  numeric=(jp-jm)/(hi-lo)/.5
  S,D,R=mid[k];ls,ld,lr=lmid[k]
  analytical=2*w[2]*u[k]-(ls*(m.P0[3]+m.P0[7])*S+ld*(m.P0[4]*D-m.P0[7]*S)+lr*m.P0[5]*R)
  gr={'case':fn.stem,'interval':k,'u':u[k],'grad_J_per_dt_num':float(numeric),'grad_hamiltonian':float(analytical),'absdiff':float(abs(numeric-analytical))};grad_rows.append(gr)
 if (s,ini) in [('A','AT'),('A','half'),('B','AT'),('C','MTD')]:
  jd,yd=m.dop853_eval(u,.5,m.P0,w)
  dc={'case':fn.stem,'J_rk4_augmented':J,'J_DOP853_augmented':float(jd),'J_abs_diff':float(abs(jd-J)),'terminal_state_max_diff':float(np.max(abs(yd[:3]-y[-1,:3]))),'A_u2_diff':float(abs(yd[6]-y[-1,6]))};dop_rows.append(dc);print('DOP',dc,flush=True)
for name,rr in [('reverified_baseline.csv',rows),('gradient_spotcheck.csv',grad_rows),('independent_dop853.csv',dop_rows)]:
 with open(out/name,'w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
print('GRAD_MAX_ABS',max(r['absdiff'] for r in grad_rows))
