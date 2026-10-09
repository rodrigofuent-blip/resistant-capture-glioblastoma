import numpy as np,csv
from pathlib import Path
from control_dynamics import dop853_eval,objective,forward,P0,IC,W
from control_discrete_adjoint import exact_discrete_gradient
b=Path(__file__).resolve().parents[1]
names=['A_AT_dt0.5_safe0_exact','A_half_dt0.5_safe0_exact_continued','A_MTD_dt0.5_safe0_exact_continued','B_AT_dt0.5_safe0_exact','C_MTD_dt0.5_safe0_exact','B_half_dt0.5_safe1000_exact','A_AT_dt0.25_safe0_exact_warm','A_half_dt0.25_safe0_exact_warm','B_AT_dt0.1_safe0_exact','C_MTD_dt0.1_safe0_exact']
out=Path(__file__).resolve().parents[3]/'work/control_validation'
out.mkdir(parents=True,exist_ok=True)
rows=[]
for name in names:
 z=np.load(Path(__file__).resolve().parents[3]/'data/control'/f'{name}.npz');u=z['u'];y=z['y'];dt=float(name.split('_dt')[1].split('_')[0]);scene=name[0];ws=1000 if 'safe1000' in name else 0
 w=W[scene].copy();w[7]=ws;w[8]=.52
 p=P0.copy();J=float(objective(y,w));JD,yd=dop853_eval(u,dt,p,w)
 yy,_=forward(u,dt,p,w,IC)
 g=exact_discrete_gradient(y,u,dt,p,w);res=float(np.max(abs(np.clip(u-g/(2*w[2]*dt),0,1)-u)))
 row=dict(case=name,intervals=len(u),time_final=len(u)*dt,RK4_objective=J,DOP853_objective=JD,absolute_J_difference=abs(J-JD),forward_reproduction_maxabs=float(np.max(abs(yy-y))),final_state_difference=float(np.max(abs(yd[:3]-y[-1,:3]))),control_exposure_sum=float(np.sum(u)*dt),control_exposure_integrated=float(y[-1,5]),control_exposure_absdiff=float(abs(np.sum(u)*dt-y[-1,5])),safety_exposure=float(y[-1,7]),min_state=float(np.min(y[:,:3])),discrete_residual=res)
 rows.append(row); print('VALIDATE',row,flush=True)
 assert row['forward_reproduction_maxabs']<1e-10
 assert row['control_exposure_absdiff']<1e-9
 assert row['time_final']==300.
 assert row['min_state']>=-1e-10
 assert row['absolute_J_difference']<.002
with open(out/'validation_summary.csv','w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
print('VALIDATION_PASS',len(rows),'max_DOP853_J_difference',max(r['absolute_J_difference'] for r in rows),'max_exposure_diff',max(r['control_exposure_absdiff'] for r in rows))
