#!/usr/bin/env python3
"""Reproduce the temporal switching table for the representative T=1500 case."""
from pathlib import Path
from output_safety import assert_writable_output
import argparse, csv, sys
from paths import ROOT, GRID_DATA, CONTROL_DATA, SENSITIVITY_DATA
sys.path.insert(0,str(ROOT/'src/event_grids'))
from event_core import BASE, archived_node_sim, event_sim

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',type=Path,default=ROOT/'work');a=ap.parse_args();assert_writable_output(a.outdir);a.outdir.mkdir(parents=True,exist_ok=True)
    p=BASE.copy();p[11]=0.55;p[12]=0.45
    rows=[]
    for dt in [0.50,0.25,0.10,0.05,0.02,0.01]:
        y=archived_node_sim(p,0.5,0.0,0.02,1500.0,dt)
        rows.append(dict(method='node',dt=dt,phi_R_T=float(y[0]),A_R=float(y[1]),A_u=float(y[3]),activations=int(y[5])))
    y=event_sim(p,0.5,0.0,0.02,1500.0,0.1,1e-11)
    rows.append(dict(method='event_localized',dt='',phi_R_T=float(y[0]),A_R=float(y[1]),A_u=float(y[3]),activations=int(y[5])))
    path=a.outdir/'temporal_switching_table.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(path)
if __name__=='__main__':main()
