#!/usr/bin/env python3
"""Recompute event-localized parameter grids from the published model.

Reference matrices are read-only; fresh outputs are written to a separate directory.
The sampled mode checks three selected cells in each grid. Full mode processes
the 50x50, 25x25 and 60x60 grids, comparing every computed cell with the archive.
"""
import argparse,sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'event_prcc'/'core'))
from event_core import BASE,COLS,event_sim,mtd_sim
NAMES=['phi_R_T','A_R','A_N','A_u','N_T','n_activations','n_deactivations','first_on','t_final','S_T','D_T','R_T','final_on']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True,help='Directory containing gridNN_event_*.csv reference arrays');ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--full',action='store_true');args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
 checks=[]
 for n,initial,T,on,off in [(50,(None,.005,None),1000.,.35,.20),(25,(.20,.005,0.),1000.,.35,.20),(60,(.50,0.,.02),1500.,.55,.45)]:
  a=args.reference;axis0=np.loadtxt(a/(f'grid{n}_R0.csv' if n==50 else f'grid{n}_alpha1.csv'));axis1=np.loadtxt(a/(f'grid{n}_S0.csv' if n==50 else f'grid{n}_theta.csv'))
  indices=[(0,0),(2,min(11,n-1)),(n-1,n-1)] if not args.full else [(i,j) for i in range(n) for j in range(n)]
  arr={k:np.full((n,n),np.nan) for k in NAMES};mtd={k:np.full((n,n),np.nan) for k in ['phi_R_T','A_R','A_N','A_u','N_T']}
  for i,j in indices:
   p=BASE.copy();p[11]=on;p[12]=off
   if n==50:S0=float(axis1[j]);D0=.005;R0=float(axis0[i])
   else:S0,D0,R0=initial;p[7]=float(axis0[i]);p[9]=p[8]*float(axis1[j])
   y=event_sim(p,S0,D0,R0,T,.1)
   assert np.isfinite(y).all() or (np.isnan(y[7]) and np.isfinite(np.delete(y,7)).all())
   for k,v in zip(NAMES,y):arr[k][i,j]=v
   if n==50 and args.full:
    z=mtd_sim(p,S0,D0,R0,T,.1)
    for key,offset in [('phi_R_T',0),('A_R',1),('A_N',2),('A_u',3),('N_T',4)]:mtd[key][i,j]=z[offset]
   for key in ['phi_R_T','A_R','A_N','A_u','N_T','n_activations','t_final']:
    refpath=a/f'grid{n}_event_{key}.csv'; ref=np.loadtxt(refpath,delimiter=',')[i,j]
    diff=abs(y[NAMES.index(key)]-ref)
    tol=1e-7 if key!='t_final' else 1e-8
    checks.append(dict(grid=n,i=i,j=j,variable=key,recomputed=y[NAMES.index(key)],reference=ref,abs_error=diff,tolerance=tol,pass_=bool(diff<=tol)))
  if args.full:
   for k,m in arr.items():np.savetxt(args.outdir/f'grid{n}_event_{k}.csv',m,delimiter=',',fmt='%.17g')
   if n==50:
    for k,m in mtd.items():np.savetxt(args.outdir/f'grid50_MTD_{k}.csv',m,delimiter=',',fmt='%.17g')
  print(f'grid {n}: {len(indices)} points verified; max-error:',max(x['abs_error'] for x in checks if x['grid']==n),flush=True)
 import csv
 with (args.outdir/'GRID_REEXECUTED_CHECKS.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=checks[0]);wr.writeheader();wr.writerows(checks)
 total=len(checks);passing=sum(x['pass_'] for x in checks)
 print('GRID_CHECK_SUMMARY',json.dumps(dict(mode='full' if args.full else 'smoke',comparisons=total,passed=passing,failed=total-passing)))
 if passing!=total:raise SystemExit('GRID MISMATCH: inspect GRID_REEXECUTED_CHECKS.csv')
if __name__=='__main__':main()
