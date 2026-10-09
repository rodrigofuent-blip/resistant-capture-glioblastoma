#!/usr/bin/env python3
"""Run a single adaptive, continuous-treatment, or untreated trajectory with command-line parameter values."""
from pathlib import Path
from output_safety import assert_writable_output
import argparse, json, sys
from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA
sys.path.insert(0,str(ROOT/'src/event_grids'))
from event_core import BASE,event_sim,mtd_sim,rk,COLS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--therapy',choices=['adaptive','mtd','off'],default='adaptive')
    p.add_argument('--S0',type=float,default=.5);p.add_argument('--D0',type=float,default=0.0);p.add_argument('--R0',type=float,default=.02);p.add_argument('--T',type=float,default=1500.0)
    p.add_argument('--rS',type=float);p.add_argument('--rD',type=float);p.add_argument('--rR',type=float);p.add_argument('--dS',type=float);p.add_argument('--dD',type=float);p.add_argument('--dR',type=float)
    p.add_argument('--alpha0',type=float);p.add_argument('--alpha1',type=float);p.add_argument('--beta',type=float);p.add_argument('--eta',type=float);p.add_argument('--K',type=float)
    p.add_argument('--N_on',type=float);p.add_argument('--N_off',type=float);p.add_argument('--hmax',type=float,default=.1);p.add_argument('--event-tol',type=float,default=1e-11)
    p.add_argument('--out',type=Path,default=ROOT/'work/custom_trajectory.json')
    a=p.parse_args();par=BASE.copy(); names=['rS','rD','rR','dS','dD','dR','alpha0','alpha1','beta','eta','K','N_on','N_off']
    for i,n in enumerate(names):
        v=getattr(a,n)
        if v is not None: par[i]=v
    if a.therapy=='adaptive':
        y=event_sim(par,a.S0,a.D0,a.R0,a.T,a.hmax,a.event_tol);result=dict(zip(COLS,map(float,y)))
    elif a.therapy=='mtd':
        y=mtd_sim(par,a.S0,a.D0,a.R0,a.T,a.hmax); result=dict(phi_R_T=float(y[0]),A_R=float(y[1]),A_N=float(y[2]),A_u=float(y[3]),N_T=float(y[4]),t_final=float(y[5]),S_T=float(y[6]),D_T=float(y[7]),R_T=float(y[8]))
    else:
        S,D,R=a.S0,a.D0,a.R0;AR=AN=AU=t=0.0
        while t<a.T-1e-13:
            h=min(a.hmax,a.T-t);S,D,R,AR,AN,AU=rk(S,D,R,AR,AN,AU,0.0,h,par);t+=h
        N=S+D+R;result=dict(phi_R_T=R/N,A_R=AR,A_N=AN,A_u=AU,N_T=N,t_final=t,S_T=S,D_T=D,R_T=R)
    payload={'therapy':a.therapy,'parameters':dict(zip(names,map(float,par))),'initial_condition':{'S0':a.S0,'D0':a.D0,'R0':a.R0},'T':a.T,'result':result}
    assert_writable_output(a.out);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
