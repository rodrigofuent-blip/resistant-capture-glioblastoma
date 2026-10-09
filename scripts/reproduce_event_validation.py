#!/usr/bin/env python3
"""Reproduce the five event-localized validation cases reported in the Supplementary Information.

Uses the event-localized ODE core (RK4 h<=0.1 with bisection event localization
at 1e-11 model-time units) and an independent SciPy DOP853 event implementation.
The validation points, tolerances, and diagnostics match the reported numerical protocol.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from output_safety import assert_writable_output
import numpy as np
from scipy.integrate import solve_ivp

from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA
sys.path.insert(0, str(ROOT/'src/event_grids'))
from event_core import BASE, event_sim  # noqa: E402

IC=np.array([0.5,0.0,0.02],float); T=1500.0
CASES=[
 ('baseline',0.05,0.356),
 ('R1_low',0.016932702245413263,0.7966101694915254),
 ('R3_moderate',0.10148353690245611,0.7966101694915254),
 ('R4_capture',0.4926907508954772,0.5084745762711864),
 ('R5_dominance',1.7438834705720385,0.6101694915254238),
]

def params(theta,alpha1):
    p=BASE.copy(); p[7]=alpha1; p[9]=theta*p[8]; return p

def rhs_aug(t,y,u,p):
    S,D,R,AR,AN,AU=y; N=S+D+R; a=p[6]+p[7]*u; q=1-N/p[10]
    return np.array([p[0]*S*q-p[3]*u*S-a*S+p[8]*D,
                     p[1]*D*q-p[4]*u*D+a*S-(p[8]+p[9])*D,
                     p[2]*R*q-p[5]*u*R+p[9]*D,
                     R,N,u],float)

def dop853_event(p,max_step=2.0,rtol=1e-11,atol=1e-13):
    y=np.r_[IC,0.,0.,0.]; t=0.; on=bool(np.sum(IC)>=p[11]); n_on=n_off=0; first=np.nan
    guard=0
    while t < T-1e-12:
        guard += 1
        if guard>1000: raise RuntimeError('event-loop guard triggered')
        u=1.0 if on else 0.0; thr=p[12] if on else p[11]; direction=-1.0 if on else 1.0
        def ev(tt,z): return (z[0]+z[1]+z[2]-thr)*direction
        ev.terminal=True; ev.direction=1.0
        sol=solve_ivp(lambda tt,z: rhs_aug(tt,z,u,p),(t,T),y,method='DOP853',
                      rtol=rtol,atol=atol,max_step=max_step,events=ev)
        if not sol.success: raise RuntimeError(sol.message)
        y=sol.y[:,-1]; t=float(sol.t[-1])
        if sol.t_events[0].size==0: break
        on=not on
        if on:
            n_on+=1
            if np.isnan(first): first=t
        else: n_off+=1
        # Advance infinitesimally on the new branch to avoid immediate re-detection.
        eps=min(1e-10,T-t)
        if eps>0:
            sol2=solve_ivp(lambda tt,z: rhs_aug(tt,z,1.0 if on else 0.0,p),(t,t+eps),y,
                           method='DOP853',rtol=rtol,atol=atol,max_step=eps)
            y=sol2.y[:,-1]; t=float(sol2.t[-1])
    S,D,R,AR,AN,AU=y; N=S+D+R
    return dict(phi_R_T=R/N,A_R=AR,A_N=AN,A_u=AU,N_T=N,first_on=first,
                n_activations=n_on,n_deactivations=n_off,t_final=t,S_T=S,D_T=D,R_T=R,final_on=int(on))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',type=Path,default=ROOT/'work'); args=ap.parse_args(); assert_writable_output(args.outdir); args.outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; tight_rows=[]
    maxima={k:0. for k in ['delta_phi','delta_AR','delta_AN','delta_Au','delta_NT','delta_first_on']}
    for name,theta,a1 in CASES:
        p=params(theta,a1)
        ev=event_sim(p,*IC,T,.1,1e-11)
        rk=dict(zip(['phi_R_T','A_R','A_N','A_u','N_T','n_activations','n_deactivations','first_on','t_final','S_T','D_T','R_T','final_on'],map(float,ev)))
        dp=dop853_event(p,max_step=2.0)
        dp025=dop853_event(p,max_step=.25)
        d={
          'delta_phi':rk['phi_R_T']-dp['phi_R_T'], 'delta_AR':rk['A_R']-dp['A_R'],
          'delta_AN':rk['A_N']-dp['A_N'], 'delta_Au':rk['A_u']-dp['A_u'],
          'delta_NT':rk['N_T']-dp['N_T'], 'delta_first_on':rk['first_on']-dp['first_on']}
        for k,v in d.items(): maxima[k]=max(maxima[k],abs(float(v)))
        rows.append({'point':name,'Theta':theta,'alpha1':a1,**d,
                     'event_cycles':rk['n_activations'],'DOP853_cycles':dp['n_activations'],
                     'event_first_on':rk['first_on'],'DOP853_first_on':dp['first_on']})
        for key in ['phi_R_T','A_R','A_N','A_u','N_T','first_on']:
            tight_rows.append({'point':name,'quantity':key,'DOP853_max_step_2':dp[key],
                               'DOP853_max_step_0.25':dp025[key],
                               'abs_difference':abs(dp[key]-dp025[key])})
    with (args.outdir/'event_validation_five_cases.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (args.outdir/'event_validation_maxstep_crosscheck.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(tight_rows[0]));w.writeheader();w.writerows(tight_rows)
    archived=GRID_DATA/'event_core_vs_DOP853_crosscheck.csv'
    archived_rows=list(csv.DictReader(archived.open(encoding='utf8')))
    # Compare the six reported differences with the packaged reference record.
    archive_max=0.0
    for a,b in zip(rows,archived_rows):
        if a['point']!=b['point']: raise SystemExit('Archived event-validation point order mismatch')
        for k in ['delta_phi','delta_AR','delta_AN','delta_Au','delta_NT','delta_first_on']:
            archive_max=max(archive_max,abs(float(a[k])-float(b[k])))
    summary={'cases':5,'production_hmax':0.1,'production_bisection_tolerance':1e-11,
             'DOP853_rtol':1e-11,'DOP853_atol':1e-13,'DOP853_max_step':2.0,
             'DOP853_secondary_max_step':0.25,'max_abs_discrepancies':maxima,
             'max_abs_difference_vs_archived_crosscheck':archive_max,
             'activation_counts_agree':all(r['event_cycles']==r['DOP853_cycles'] for r in rows)}
    (args.outdir/'event_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    if not summary['activation_counts_agree']: raise SystemExit('Activation count mismatch')
    if archive_max>5e-10: raise SystemExit(f'Event validation differs from the packaged reference cross-check: {archive_max:g}')

if __name__=='__main__': main()
