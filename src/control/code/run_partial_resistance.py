import time,csv,numpy as np
from pathlib import Path
from control_discrete_adjoint import solve_exact
b=Path(__file__).resolve().parents[1];rows=[];out=Path(__file__).resolve().parents[3]/'work/standalone_partial_resistance';out.mkdir(parents=True,exist_ok=True)
for dr in [.005,.0055]:
 for ini in ['AT','half','MTD']:
  t=time.monotonic();z=solve_exact('A',ini,.5,max_iter=1000,dr_ratio=dr)
  d=z['y'][-1];row={k:z[k] for k in ['scenario','init','dt','wsafe','nsafe','dr_ratio','best_J','last_J','iterations','best_iteration','best_residual','last_residual','converged']}
  row|={'A_u':d[5],'A_R':d[4],'phi_R_T':d[2]/sum(d[:3]),'elapsed_s':time.monotonic()-t}
  key=f'A_dr{dr:g}_{ini}_dt0.5_exact';np.savez_compressed(out/f'{key}.npz',u=z['u'],y=z['y'],u_last=z['u_last'],y_last=z['y_last'],hist_J=z['hist_J'],hist_resid=z['hist_R'],hist_step=z['hist_step'])
  rows.append(row);print('PARTIAL',key,row['best_J'],row['best_residual'],row['last_residual'],row['converged'],row['elapsed_s'],flush=True)
  with open(out/'partial_resistance_exact.csv','w',newline='') as f:
   wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
