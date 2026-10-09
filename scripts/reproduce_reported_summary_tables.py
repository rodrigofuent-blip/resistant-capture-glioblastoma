#!/usr/bin/env python3
"""Regenerate numerical summary tables reported in the Main manuscript and Supplementary Information.

The script reads the packaged event-grid, event-PRCC, and exact-discrete control outputs.
Reported values are calculated from those numerical files rather than copied from the manuscript.
"""
from pathlib import Path
from output_safety import assert_writable_output
import argparse, csv, json, sys
import numpy as np
import pandas as pd
from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA
sys.path.insert(0,str(ROOT/'src/control/code'))
from control_dynamics import W, objective  # noqa:E402

def arr(n,key): return np.loadtxt(GRID_DATA/f'grid{n}_event_{key}.csv',delimiter=',')
def mtd(key): return np.loadtxt(GRID_DATA/f'grid50_MTD_{key}.csv',delimiter=',')
def npz_metrics(path,scenario,wsafe=0.,nsafe=.52):
    z=np.load(path); y=z['y']; Y=y[-1]; N=Y[:3].sum(); R=Y[2]; w=W[scenario].copy();w[7]=wsafe;w[8]=nsafe
    # exact residual from archived histories is last history value only for converged best=last cases;
    # reported residual values are represented in the packaged check tables below.
    return dict(J=float(objective(y,w)),A_N=float(Y[3]),A_R=float(Y[4]),A_u=float(Y[5]),N_T=float(N),R_T=float(R),phi_R_T=float(R/N),iterations=len(z['hist_J']))

def write_csv(path,rows):
    rows=list(rows)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',type=Path,default=ROOT/'work/reported_tables');a=ap.parse_args();assert_writable_output(a.outdir);a.outdir.mkdir(parents=True,exist_ok=True)
    phi,AR,Au,cyc=arr(50,'phi_R_T'),arr(50,'A_R'),arr(50,'A_u'),arr(50,'n_activations')
    cycle=[]
    for k in sorted(np.unique(cyc).astype(int)):
        m=cyc==k;cycle.append({'n_cycles':k,'fraction_of_grid':float(m.mean()),'mean_phi_R_T':float(phi[m].mean()),'mean_A_u':float(Au[m].mean()),'count':int(m.sum())})
    write_csv(a.outdir/'main_cycle_stratified_summary.csv',cycle)
    ARm,phim,Aum=mtd('A_R'),mtd('phi_R_T'),mtd('A_u')
    write_csv(a.outdir/'main_AT_vs_MTD_summary.csv',[{
      'AT_mean_phi_R_T':float(phi.mean()),'MTD_mean_phi_R_T':float(phim.mean()),
      'AT_mean_A_R':float(AR.mean()),'MTD_mean_A_R':float(ARm.mean()),
      'AT_mean_A_u':float(Au.mean()),'MTD_mean_A_u':float(Aum.mean()),
      'fraction_phi_AT_lower':float(np.mean(phi<phim)),'fraction_AR_AT_lower':float(np.mean(AR<ARm)),
      'mean_AR_ratio_AT_over_MTD':float(np.mean(AR/ARm))}])
    p25,ar25=arr(25,'phi_R_T'),arr(25,'A_R')
    write_csv(a.outdir/'main_parametric_25x25_summary.csv',[{
      'grid_rows':25,'grid_cols':25,'min_A_R':float(ar25.min()),'max_A_R':float(ar25.max()),
      'min_phi_R_T':float(p25.min()),'max_phi_R_T':float(p25.max()),
      'fraction_phi_ge_0p2':float(np.mean(p25>=.2)),'fraction_phi_ge_0p5':float(np.mean(p25>=.5))}])
    write_csv(a.outdir/'supp_threshold_sensitivity.csv',[{'phi_c':x,'fraction_high_risk':float(np.mean(phi>=x))} for x in [.1,.2,.3,.5]])
    lhs=pd.read_csv(SENSITIVITY_DATA/'lhs_results_event_2000.csv'); pr=pd.read_csv(SENSITIVITY_DATA/'prcc_event_15_predictors_activation_subset.csv')
    write_csv(a.outdir/'supp_global_sensitivity_counts.csv',[{'n_LHS':len(lhs),'n_activation_subset':int(lhs.first_on.notna().sum()),'n_failed':int(lhs.failed.sum())}])
    agg=pr.groupby('parameter',as_index=False)['abs_PRCC'].mean().sort_values('abs_PRCC',ascending=False)
    agg.to_csv(a.outdir/'supp_mean_absolute_PRCC.csv',index=False)
    # Stationarity is evaluated at the returned control; iteration histories are retained
    # as a distinct, pre-update stopping diagnostic.
    primary=pd.concat([pd.read_csv(CONTROL_DATA/'partial_resistance_exact.csv'),pd.read_csv(CONTROL_DATA/'partial_resistance_refined_exact.csv')],ignore_index=True)
    primary_idx={(float(row.dr_ratio),float(row.dt),str(row.init)):row for row in primary.itertuples(index=False)}
    partial=[]
    for rho in [.005,.0055]:
      for dt,init in [(0.5,'AT'),(0.5,'half'),(0.5,'MTD'),(0.25,'AT'),(0.25,'half')]:
        fn=f'A_dr{rho:g}_{init}_dt{dt}_exact.npz'
        x=npz_metrics(CONTROL_DATA/fn,'A')
        with np.load(CONTROL_DATA/fn) as z:
          iteration_residual=float(z['hist_resid'][-1])
        final_residual=float(primary_idx[(rho,dt,init)].best_residual)
        partial.append({'rho_R':rho,'dt':dt,'initialization':init,**x,
              'returned_control_projected_residual':final_residual,
              'pre_update_iteration_residual':iteration_residual})
    write_csv(a.outdir/'supp_partial_resistant_sensitivity.csv',partial)
    safety=[]
    for dt,init,fn in [(.5,'AT','B_AT_dt0.5_safe1000_exact.npz'),(.5,'half','B_half_dt0.5_safe1000_exact.npz'),(.5,'MTD','B_MTD_dt0.5_safe1000_exact.npz'),(.25,'half','B_half_dt0.25_safe1000_exact.npz')]:
        x=npz_metrics(CONTROL_DATA/fn,'B',1000.,.52); z=np.load(CONTROL_DATA/fn)
        safety.append({'dt':dt,'initialization':init,**x,'best_observed_residual':float(np.min(z['hist_resid'])),'last_residual':float(z['hist_resid'][-1])})
    write_csv(a.outdir/'supp_soft_safety_summary.csv',safety)
    # Include the exact-discrete stationarity check table when available.
    src=ROOT/'data/control_checks/refinement_summary.csv'
    if src.exists(): pd.read_csv(src).to_csv(a.outdir/'control_refinement_source_check.csv',index=False)
    summary={'generated_files':sorted(p.name for p in a.outdir.glob('*.csv')),'grid50_cells':int(phi.size),'grid25_cells':int(p25.size),'LHS_rows':int(len(lhs))}
    (a.outdir/'reported_tables_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
