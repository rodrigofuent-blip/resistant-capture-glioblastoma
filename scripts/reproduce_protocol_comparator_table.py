#!/usr/bin/env python3
"""Reproduce the protocol/candidate table reported in the Supplementary Information.

The calculation uses the event-localized model core, the exact-discrete optimal-control
candidates, and the A/B/C objective weights used in the reported analysis.

Outputs:
  protocol_control_candidate_outcomes.csv
  comparator_crosscheck.csv
  protocol_comparator_summary.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from output_safety import assert_writable_output
import numpy as np
from scipy.integrate import solve_ivp

from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA
sys.path.insert(0, str(ROOT/'src/event_grids')); sys.path.insert(0, str(ROOT/'src/control/code'))
from event_core import BASE, event_sim, rk, mtd_sim  # noqa: E402
from control_dynamics import W  # noqa: E402

P = BASE.copy(); IC = np.array([0.5, 0.0, 0.02]); T = 300.0

def objective_from_metrics(AN, AR, Au2, N, R, w):
    return float(w[0]*AN + w[1]*AR + w[2]*Au2 + w[3]*N + w[4]*R + w[5]*R/(N+w[6]))

def row_with_costs(name, AN, AR, Au, N, R, phi, Au2=None):
    if Au2 is None: Au2 = Au
    row = dict(protocol=name, A_N=float(AN), A_R=float(AR), A_u=float(Au), N_T=float(N), R_T=float(R), phi_R_T=float(phi))
    for s in 'ABC': row['J_'+s] = objective_from_metrics(AN, AR, Au2, N, R, W[s])
    return row

def rk_periodic(on, off, h=0.05):
    S,D,R = IC; AR=AN=AU=AU2=0.0; n=int(round(T/h)); period=on+off
    for k in range(n):
        t=k*h; phase=t-period*np.floor(t/period+1e-12); u=1.0 if phase < on-1e-12 else 0.0
        S,D,R,AR,AN,AU = rk(S,D,R,AR,AN,AU,u,h,P); AU2 += u*u*h
    N=S+D+R
    return row_with_costs('',AN,AR,AU,N,R,R/N,AU2)

def rhs_aug(t,y,u):
    S,D,R,AR,AN,AU=y; N=S+D+R; a=P[6]+P[7]*u; q=1-N/P[10]
    return [P[0]*S*q-P[3]*u*S-a*S+P[8]*D,
            P[1]*D*q-P[4]*u*D+a*S-(P[8]+P[9])*D,
            P[2]*R*q-P[5]*u*R+P[9]*D,
            R,N,u]

def dop_fixed_intervals(intervals,rtol=1e-9,atol=1e-11,max_step=2.0):
    y=np.r_[IC,0.,0.,0.]
    for a,b,u in intervals:
        if b<=a: continue
        sol=solve_ivp(lambda t,z:rhs_aug(t,z,u),(a,b),y,method='DOP853',rtol=rtol,atol=atol,max_step=max_step)
        if not sol.success: raise RuntimeError(sol.message)
        y=sol.y[:,-1]
    S,D,R,AR,AN,AU=y; N=S+D+R
    return np.array([AN,AR,AU,N,R,R/N])

def periodic_intervals(on,off):
    intervals=[]; period=on+off; q=0
    while q*period < T-1e-14:
        a=q*period; b=min(a+on,T); c=min(a+period,T)
        if b>a: intervals.append((a,b,1.0))
        if c>b: intervals.append((b,c,0.0))
        q+=1
    return intervals

def dop_event(rtol=1e-9,atol=1e-11,max_step=2.0):
    y=np.r_[IC,0.,0.,0.]; t=0.0; on=bool(np.sum(IC)>=P[11])
    while t < T-1e-12:
        u=1.0 if on else 0.0; thr=P[12] if on else P[11]; direction=-1.0 if on else 1.0
        def ev(tt,z): return (z[0]+z[1]+z[2]-thr)*direction
        ev.terminal=True; ev.direction=1
        sol=solve_ivp(lambda tt,z:rhs_aug(tt,z,u),(t,T),y,method='DOP853',rtol=rtol,atol=atol,max_step=max_step,events=ev)
        if not sol.success: raise RuntimeError(sol.message)
        y=sol.y[:,-1]; t=float(sol.t[-1])
        if sol.t_events[0].size==0: break
        on=not on
    S,D,R,AR,AN,AU=y; N=S+D+R
    return np.array([AN,AR,AU,N,R,R/N])

def candidate_row(label,filename):
    z=np.load(CONTROL_DATA/filename,allow_pickle=False); y=z['y']; u=z['u']; dt=T/len(u); Y=y[-1]; N=float(np.sum(Y[:3])); R=float(Y[2])
    return row_with_costs(label,Y[3],Y[4],Y[5],N,R,R/N,float(np.sum(u*u)*dt))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',type=Path,default=ROOT/'work'); args=ap.parse_args(); assert_writable_output(args.outdir); args.outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; checks=[]
    # MTD: canonical event/RK4 core, h=0.1; binary u => Au2=Au
    m=mtd_sim(P,*IC,T,.1); rows.append(row_with_costs('MTD-like continuous',m[2],m[1],m[3],m[4],m[8],m[0],m[3]))
    # Adaptive therapy: event-localized threshold rule
    e=event_sim(P,*IC,T,.1); rows.append(row_with_costs('Threshold AT (event localized)',e[2],e[1],e[3],e[4],e[11],e[0],e[3]))
    q=rk_periodic(14.,14.); q['protocol']='Fixed intermittent 14/14'; rows.append(q)
    q=rk_periodic(5.,23.); q['protocol']='Normalized Stupp-like 5/23'; rows.append(q)
    rows += [candidate_row('A--delayed candidate','A_AT_dt0.5_safe0_exact.npz'),
             candidate_row('A--early candidate','A_half_dt0.5_safe0_exact_continued.npz'),
             candidate_row('B near-zero candidate','B_AT_dt0.5_safe0_exact.npz'),
             candidate_row('C early-taper candidate','C_MTD_dt0.5_safe0_exact.npz')]
    # Independent DOP853 comparator checks. These tolerances are for this protocol-table cross-check,
    # distinct from the tighter five-case event validation reported elsewhere in the Supplement.
    rkvals={r['protocol']:np.array([r['A_N'],r['A_R'],r['A_u'],r['N_T'],r['R_T'],r['phi_R_T']]) for r in rows[:4]}
    dopvals={
      'MTD-like continuous':dop_fixed_intervals([(0,T,1.0)]),
      'Threshold AT (event localized)':dop_event(),
      'Fixed intermittent 14/14':dop_fixed_intervals(periodic_intervals(14.,14.)),
      'Normalized Stupp-like 5/23':dop_fixed_intervals(periodic_intervals(5.,23.)),
    }
    labels=['A_N','A_R','A_u','N_T','R_T','phi_R_T']; maxdiff=0.0
    for name in dopvals:
        for j,lbl in enumerate(labels):
            d=float(abs(rkvals[name][j]-dopvals[name][j])); maxdiff=max(maxdiff,d)
            checks.append(dict(protocol=name,quantity=lbl,RK4=float(rkvals[name][j]),DOP853=float(dopvals[name][j]),abs_difference=d))
    with (args.outdir/'protocol_control_candidate_outcomes.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (args.outdir/'comparator_crosscheck.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0])); w.writeheader(); w.writerows(checks)
    summary={'rows':len(rows),'comparator_rows':4,'candidate_rows':4,'max_cross_method_abs_difference':maxdiff,
             'rounded_scientific_notation_3sf':f'{maxdiff:.2e}','event_AT_hmax':0.1,'periodic_RK4_h':0.05,
             'DOP853_rtol':1e-9,'DOP853_atol':1e-11,'DOP853_max_step':2.0}
    (args.outdir/'protocol_comparator_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    # The reported value is 2.26e-7 after rounding; accept a narrow numerical band.
    if not (2.20e-7 <= maxdiff <= 2.30e-7): raise SystemExit('Comparator cross-check outside expected range')
if __name__=='__main__': main()
