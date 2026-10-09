#!/usr/bin/env python3
"""Reproduce the exact-discrete optimal-control cases reported in Paper 1.

The script writes all generated files under --outdir. It reproduces the A/B/C candidates,
the Scenario-A continuation and mesh-refinement calculations, the partial-resistant-sensitivity
rows, and the soft burden-safety rows reported in the Supplementary Information.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
from output_safety import assert_writable_output
import numpy as np
from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA
sys.path.insert(0,str(ROOT/'src/control/code'))
from control_discrete_adjoint import solve_exact  # noqa: E402
from control_dynamics import W, objective  # noqa: E402

def save_npz(path,r):
    np.savez_compressed(path,u=r['u'],y=r['y'],u_last=r['u_last'],y_last=r['y_last'],hist_J=r['hist_J'],hist_resid=r['hist_R'],hist_step=r['hist_step'])

def metrics(tag,r):
    y=r['y']; Y=y[-1]; N=float(Y[:3].sum()); R=float(Y[2]);
    return {'case':tag,'scenario':r['scenario'],'init':r['init'],'dt':r['dt'],'dr_ratio':r['dr_ratio'],'wsafe':r['wsafe'],
            'iterations':r['iterations'],'best_iteration':r['best_iteration'],'best_J':r['best_J'],'best_residual':r['best_residual'],
            'last_residual':r['last_residual'],'converged':bool(r['converged']),'A_N':float(Y[3]),'A_R':float(Y[4]),'A_u':float(Y[5]),
            'A_u2':float(Y[6]),'A_safe':float(Y[7]),'N_T':N,'R_T':R,'phi_R_T':R/N}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',type=Path,default=ROOT/'work/control_reexecution');
    ap.add_argument('--quick',action='store_true',help='Run baseline A/B/C + A refinement only; omit partial-resistance and safety scans.')
    args=ap.parse_args(); assert_writable_output(args.outdir); args.outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; start=time.time()
    def run(tag,*a,**kw):
        print('RUN',tag,flush=True); r=solve_exact(*a,**kw); save_npz(args.outdir/(tag+'.npz'),r); rows.append(metrics(tag,r)); return r
    A_del=run('A_delayed_dt0p5','A','AT',.5,max_iter=1000)
    A_early0=run('A_early_initial_1000','A','half',.5,max_iter=1000)
    A_early=run('A_early_continuation_312','A','half',.5,max_iter=312,seed_control=A_early0['u_last'])
    # Fine-mesh continuation from the retained coarse controls, as described in the Supplementary Information.
    run('A_delayed_fine_reoptimized_1200','A','AT',.25,max_iter=1200,seed_control=np.repeat(A_del['u'],2))
    run('A_early_fine_reoptimized_1200','A','half',.25,max_iter=1200,seed_control=np.repeat(A_early['u'],2))
    run('B_near_zero_dt0p5','B','AT',.5,max_iter=1000)
    run('B_near_zero_dt0p1','B','AT',.1,max_iter=1000)
    run('C_early_taper_dt0p5','C','MTD',.5,max_iter=1000)
    run('C_early_taper_dt0p1','C','MTD',.1,max_iter=1000)
    if not args.quick:
        for rho in [0.005,0.0055]:
            for init in ['AT','half','MTD']:
                run(f'A_partial_rho{rho:g}_{init}_dt0p5','A',init,.5,max_iter=1000,dr_ratio=rho)
            for init in ['AT','half']:
                run(f'A_partial_rho{rho:g}_{init}_dt0p25','A',init,.25,max_iter=1000,dr_ratio=rho)
        # Soft-safety rows. Best observed iterate is reported even when stationarity tolerance is not met.
        for init in ['AT','half','MTD']:
            run(f'B_softsafe_{init}_dt0p5','B',init,.5,max_iter=1000,wsafe=1000.,nsafe=.52)
        run('B_softsafe_half_dt0p25','B','half',.25,max_iter=650,wsafe=1000.,nsafe=.52)
    with (args.outdir/'control_reexecution_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Compare reported baseline/refinement metrics with the packaged reference NPZ files.
    reference={
      'A_delayed_dt0p5':'A_AT_dt0.5_safe0_exact.npz',
      'A_early_continuation_312':'A_half_dt0.5_safe0_exact_continued.npz',
      'A_delayed_fine_reoptimized_1200':'A_AT_dt0.25_safe0_exact_warm.npz',
      'A_early_fine_reoptimized_1200':'A_half_dt0.25_safe0_exact_warm.npz',
      'B_near_zero_dt0p5':'B_AT_dt0.5_safe0_exact.npz','B_near_zero_dt0p1':'B_AT_dt0.1_safe0_exact.npz',
      'C_early_taper_dt0p5':'C_MTD_dt0.5_safe0_exact.npz','C_early_taper_dt0p1':'C_MTD_dt0.1_safe0_exact.npz'}
    if not args.quick:
        # Match each partial-resistance and soft-burden-safety run to its archived trajectory.
        for rho in (0.005,0.0055):
            for init in ('AT','half','MTD'):
                reference[f'A_partial_rho{rho:g}_{init}_dt0p5']=f'A_dr{rho:g}_{init}_dt0.5_exact.npz'
            for init in ('AT','half'):
                reference[f'A_partial_rho{rho:g}_{init}_dt0p25']=f'A_dr{rho:g}_{init}_dt0.25_exact.npz'
        for init in ('AT','half','MTD'):
            reference[f'B_softsafe_{init}_dt0p5']=f'B_{init}_dt0.5_safe1000_exact.npz'
        reference['B_softsafe_half_dt0p25']='B_half_dt0.25_safe1000_exact.npz'
    max_u=max_y=0.0
    for tag,fn in reference.items():
        a=np.load(args.outdir/(tag+'.npz')); b=np.load(CONTROL_DATA/fn)
        max_u=max(max_u,float(np.max(np.abs(a['u']-b['u'])))); max_y=max(max_y,float(np.max(np.abs(a['y']-b['y']))))
    summary={'cases_run':len(rows),'archived_trajectories_compared':len(reference),'quick':args.quick,'elapsed_seconds':time.time()-start,'max_abs_u_vs_reference':max_u,'max_abs_y_vs_reference':max_y,
             'A_early_total_iterations':1000+312,'A_fine_reoptimization_iterations':1200,'stationarity_tolerance':1e-5,'relaxation_omega':0.05}
    (args.outdir/'control_reexecution_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    if max_u>5e-12 or max_y>5e-12: raise SystemExit('Control calculation does not match the packaged reference outputs')
if __name__=='__main__': main()
