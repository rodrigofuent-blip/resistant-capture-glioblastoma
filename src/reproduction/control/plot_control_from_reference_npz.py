#!/usr/bin/env python3
"""Reconstruct a four-panel control diagnostic from archived discrete trajectories.

This data-derived diagnostic does not replace the typeset figure in the article.
"""
from pathlib import Path
import argparse,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 a=argparse.ArgumentParser();a.add_argument('--results',type=Path,required=True);a.add_argument('--outdir',type=Path,required=True);z=a.parse_args();z.outdir.mkdir(parents=True,exist_ok=True)
 cases=[('A delayed','A_AT_dt0.5_safe0_exact.npz'),('A early','A_half_dt0.5_safe0_exact_continued.npz'),('B near-zero','B_AT_dt0.5_safe0_exact.npz'),('C early taper','C_MTD_dt0.5_safe0_exact.npz')]
 traces={}
 for name,fn in cases:
  d=np.load(z.results/fn,allow_pickle=False);y=d['y'];u=d['u'];dt=300/len(u)
  N=np.sum(y[:,:3],axis=1); phi=np.divide(y[:,2],N,out=np.zeros_like(N),where=N>0)
  traces[name]=dict(t=np.arange(len(y))*dt,tu=np.arange(len(u))*dt,u=u,N=N,phi=phi)
  print(name,'N_T',N[-1],'phi_T',phi[-1],'A_u',np.sum(u)*dt)
 fig,ax=plt.subplots(2,2,figsize=(12,8));
 for name,v in traces.items():
  ax[0,0].step(v['tu'],v['u'],where='post',label=name);ax[0,1].plot(v['t'],v['N'],label=name);ax[1,0].plot(v['t'],v['phi'],label=name);ax[1,1].scatter(v['N'][-1],v['phi'][-1],label=name);ax[1,1].annotate(name,(v['N'][-1],v['phi'][-1]),xytext=(3,3),textcoords='offset points',fontsize=8)
 ax[0,0].set(title='A: Open-loop controls',xlabel='Model time',ylabel='u(t)');ax[0,0].legend(fontsize=8)
 ax[0,1].set(title='B: Total burden',xlabel='Model time',ylabel='N(t)')
 ax[1,0].set(title='C: Resistant fraction',xlabel='Model time',ylabel='phi_R(t)')
 ax[1,1].set(title='D: Terminal trade-off',xlabel='N(T)',ylabel='phi_R(T)')
 for x in ax.flat:x.grid(alpha=.2)
 fig.tight_layout();fig.savefig(z.outdir/'CONTROL_REPLOT_FROM_ARCHIVED_NPZ.pdf');fig.savefig(z.outdir/'CONTROL_REPLOT_FROM_ARCHIVED_NPZ.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
