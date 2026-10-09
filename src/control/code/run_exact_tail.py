import time,sys,csv,numpy as np
from pathlib import Path
from control_discrete_adjoint import solve_exact
from control_dynamics import P0,IC,W,objective,forward,dop853_eval
b=Path(__file__).resolve().parents[1]
plan=[('B','AT',.1,0.,500),('C','MTD',.1,0.,500)]+[('B',ini,.5,1000.,1000) for ini in ('AT','half','MTD')]+[('B','half',.25,1000.,650)]
out=Path(__file__).resolve().parents[3]/'work/standalone_exact_tail'
out.mkdir(parents=True,exist_ok=True)
rows=[]
for s,ini,dt,wsafe,maxit in plan:
 t=time.monotonic();z=solve_exact(s,ini,dt,max_iter=maxit,wsafe=wsafe);d=z['y'][-1];nl=d[:3].sum();r=d[2]
 row={k:z[k] for k in ['scenario','init','dt','wsafe','nsafe','dr_ratio','best_J','last_J','iterations','best_iteration','best_residual','last_residual','converged']}|{
  'A_N':d[3],'A_R':d[4],'A_u':d[5],'A_u2':d[6],'A_safe':d[7],
  'N_T':nl,'R_T':r,'phi_R_T':r/nl,'N_max_nodes':z['y'][:,:3].sum(axis=1).max(),
  'runtime_s':time.monotonic()-t}
 key=f'{s}_{ini}_dt{dt:g}_safe{wsafe:g}_exact';
 np.savez_compressed(out/f'{key}.npz',u=z['u'],y=z['y'],u_last=z['u_last'],y_last=z['y_last'],hist_J=z['hist_J'],hist_resid=z['hist_R'],hist_step=z['hist_step'])
 rows.append(row)
 with open(out/'exact_discrete_tail_summary.csv','w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 print('EXACT',key,'J',row['best_J'],'Jlast',row['last_J'],'rbest',row['best_residual'],'rlast',row['last_residual'],'iter',row['iterations'],'s',round(row['runtime_s'],1),flush=True)
