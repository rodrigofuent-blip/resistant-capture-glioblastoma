#!/usr/bin/env python3
"""Generate independent diagnostic visualizations from archived grid matrices.

The generated diagnostic plots are not asserted to reproduce the precise
typesetting or graphical design of the manuscript figures.
"""
from pathlib import Path
import argparse,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();a.outdir.mkdir(parents=True,exist_ok=True)
    maps={50:['phi_R_T','A_R','A_u','n_activations'],25:['phi_R_T','A_R','A_u','n_activations'],60:['phi_R_T','A_R','A_u','regime']}
    for n,cols in maps.items():
        fig,ax=plt.subplots(2,2,figsize=(11,9))
        for i,(col,axis) in enumerate(zip(cols,ax.flat)):
            arr=np.loadtxt(a.results/f'grid{n}_event_{col}.csv',delimiter=',')
            assert arr.shape==(n,n) and np.isfinite(arr).all()
            img=axis.imshow(arr,origin='lower',aspect='auto')
            axis.set_title(f'{n}x{n}: {col}');axis.set_xlabel('S0 index' if n==50 else 'Theta index');axis.set_ylabel('R0 index' if n==50 else 'alpha1 index')
            fig.colorbar(img,ax=axis)
        fig.tight_layout();out=a.outdir/f'INDEPENDENT_GRID{n}_DATA_VIEW.pdf';fig.savefig(out);plt.close(fig)
        print('GRID_REPLOT',n,str(out))
if __name__=='__main__':main()
