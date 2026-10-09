#!/usr/bin/env python3
"""Command-line entry point for Paper 1 numerical reproduction."""
from pathlib import Path
import argparse, subprocess, sys, shutil
from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA, WORK, REF, CONTROL_CODE, REPRODUCE, PRCC_INPUTS
PY=sys.executable

def run(args,cwd=None): print('>>>',' '.join(map(str,args)),flush=True);subprocess.run(list(map(str,args)),cwd=cwd,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','smoke','verify','gradient-check','control-dop853-check','tables','temporal','event-validation','protocol-table','control-quick','control-full','grid-smoke','grid-full','prcc-figures','prcc-full','all-full']);a=p.parse_args();WORK.mkdir(parents=True,exist_ok=True)
 if a.mode in ['preflight','smoke','verify','all-full']:
  run([PY,ROOT/'docs/verify_manifests.py'])
  run([PY,ROOT/'scripts/verify_reference_data.py'])
 if a.mode in ['smoke','tables','all-full']: run([PY,ROOT/'scripts/reproduce_reported_summary_tables.py'])
 if a.mode in ['smoke','tables','all-full']:
  run([PY,ROOT/'scripts/verify_supp_table_S1.py',WORK/'reported_tables/supp_partial_resistant_sensitivity.csv'])
 if a.mode in ['temporal','all-full']: run([PY,ROOT/'scripts/reproduce_temporal_convergence.py'])
 if a.mode in ['protocol-table','all-full']: run([PY,ROOT/'scripts/reproduce_protocol_comparator_table.py'])
 if a.mode in ['event-validation','all-full']: run([PY,ROOT/'scripts/reproduce_event_validation.py'])
 if a.mode=='control-quick': run([PY,ROOT/'scripts/reproduce_control.py','--quick','--outdir',WORK/'control_quick'])
 if a.mode in ['control-full','all-full']: run([PY,ROOT/'scripts/reproduce_control.py','--outdir',WORK/'control_full'])
 if a.mode in ['smoke','grid-smoke']: run([PY,ROOT/'src/reproduction/grids/reproduce_all_grids.py','--reference',GRID_DATA,'--outdir',WORK/'grid_smoke'])
 if a.mode in ['grid-full','all-full']: run([PY,ROOT/'src/reproduction/grids/reproduce_all_grids.py','--reference',GRID_DATA,'--outdir',WORK/'grids_full','--full'])
 if a.mode in ['gradient-check','all-full']:
  run([PY,CONTROL_CODE/'test_discrete_adjoint.py'],cwd=CONTROL_CODE)
 if a.mode in ['control-dop853-check','all-full']:
  run([PY,CONTROL_CODE/'validate_selected_exact.py'],cwd=CONTROL_CODE)
 if a.mode in ['prcc-figures','all-full']:
  target=WORK/'prcc_figures';target.mkdir(parents=True,exist_ok=True);run([PY,ROOT/'src/sensitivity/plot_prcc.py','--input-dir',SENSITIVITY_DATA/'plot_input','--output-dir',target])
 if a.mode in ['prcc-full','all-full']:
  target=WORK/'event_prcc_full'
  if target.exists():shutil.rmtree(target)
  shutil.copytree(REPRODUCE/'event_prcc',target);shutil.copytree(PRCC_INPUTS,target/'input');run([PY,target/'recompute_lhs_event_portable.py'],cwd=target)
  import hashlib
  reference=SENSITIVITY_DATA
  for name in ('lhs_results_event_2000.csv','prcc_event_15_predictors_activation_subset.csv','prcc_event_on_common_activation_subset.csv','event_top5_by_output.csv'):
   orig=reference/name; fresh=target/name
   if hashlib.sha256(orig.read_bytes()).digest()!=hashlib.sha256(fresh.read_bytes()).digest():
    raise RuntimeError('Regenerated PRCC file differs from archived reference: '+name)
  print('PRCC_REPRODUCTION_PASS: 2000 LHS rows, 120 coefficients; four archived files byte-identical',flush=True)
if __name__=='__main__':main()
