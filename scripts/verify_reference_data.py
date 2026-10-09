#!/usr/bin/env python3
"""Check that the packaged numerical reference files have the expected dimensions and key invariants."""
from pathlib import Path
import json, numpy as np, pandas as pd
from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA
checks={}
for n in (50,25,60):
    a=np.loadtxt(GRID_DATA/f'grid{n}_event_phi_R_T.csv',delimiter=',');checks[f'grid{n}_shape']=list(a.shape);assert a.shape==(n,n)
lhs=pd.read_csv(SENSITIVITY_DATA/'lhs_results_event_2000.csv');checks['LHS_rows']=len(lhs);checks['activation_subset']=int(lhs.first_on.notna().sum());checks['failed_runs']=int(lhs.failed.sum());assert len(lhs)==2000 and checks['activation_subset']==1628 and checks['failed_runs']==0
needed=['A_AT_dt0.5_safe0_exact.npz','A_half_dt0.5_safe0_exact_continued.npz','B_AT_dt0.5_safe0_exact.npz','C_MTD_dt0.5_safe0_exact.npz']
checks['control_files_present']=all((CONTROL_DATA/x).exists() for x in needed);assert checks['control_files_present']
print(json.dumps(checks,indent=2))
