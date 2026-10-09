#!/usr/bin/env python3
"""Validate Supplementary Table S1 using residuals at returned control candidates."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
# rho_R, grid step, initial control, objective (4 decimal places), exposure (3),
# projected residual in units of 1e-6 (1 decimal place).
ROWS=[
 (0.005,0.50,'AT',144.7265,8.685,9.8),
 (0.005,0.50,'half',145.1647,106.861,9.5),
 (0.005,0.50,'MTD',145.1646,106.861,9.1),
 (0.005,0.25,'AT',145.1646,106.855,9.6),
 (0.005,0.25,'half',145.1646,106.855,9.5),
 (0.0055,0.50,'AT',143.3552,112.561,9.2),
 (0.0055,0.50,'half',143.3552,112.562,9.2),
 (0.0055,0.25,'AT',143.3552,112.561,9.4),
]
def validate(csv_path):
    d=pd.read_csv(csv_path)
    for rho,dt,init,J,Au,res in ROWS:
        subset=d[(abs(d.rho_R-rho)<1e-10)&(abs(d.dt-dt)<1e-10)&(d.initialization==init)]
        assert len(subset)==1, f'Missing or duplicate Table S1 row: {(rho,dt,init)}'
        v=subset.iloc[0]
        assert abs(v.J-J)<5.1e-5, f'Objective mismatch: {(rho,dt,init)}'
        assert abs(v.A_u-Au)<5.1e-4, f'Exposure mismatch: {(rho,dt,init)}'
        assert abs(v.returned_control_projected_residual*1e6-res)<0.051, f'Residual mismatch: {(rho,dt,init)}'
        assert v.returned_control_projected_residual<1e-5, f'Stationarity tolerance failed: {(rho,dt,init)}'
    print(f'PASS Supplementary Table S1: {len(ROWS)} displayed rows; returned-control projected residual')
if __name__=='__main__':
    import sys
    validate(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/reference_tables/reported_tables/supp_partial_resistant_sensitivity.csv')
