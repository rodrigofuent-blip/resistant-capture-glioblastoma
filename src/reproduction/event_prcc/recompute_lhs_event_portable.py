#!/usr/bin/env python3
"""Event-localized LHS and PRCC computation using a fixed 2000-row input design.
The node-switched and event-localized integrations share the input matrix.
PRCCs are conditional rank associations for activation-containing trajectories.
N_max is evaluated at RK4 step endpoints and event times.
"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
from numba import njit
ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'core'))
from event_core import rk, event_sim as grid_event_sim
SAMPLE=ROOT/'input'/'lhs_samples.csv'
ARCH=ROOT/'input'/'lhs_results.csv'
PARAMS=['rS','rD','rR','dS','dD_ratio','dR','alpha0','alpha1','beta','eta','N_on','hwidth','S0','R0','T']
OUTS=['phi_R_T','A_R','A_u','A_N','N_T','N_max','n_cycles','first_on']
@njit(cache=True)
def run(p,S,D,R,T,hmax):
    AR=AN=AU=t=0.; Npeak=S+D+R; on=(Npeak>=p[11]); on0=on
    n_activations=int(on0); first_on=0. if on0 else np.nan
    n_off=0
    for outer in range(int(np.ceil(T/hmax))+40):
        if t>=T-1e-13:break
        rem=min(hmax,T-t)
        for inner in range(8):
            if rem<=1e-14:break
            u=1. if on else 0.;thr=p[12] if on else p[11];direction=-1. if on else 1.
            g0=((S+D+R)-thr)*direction
            e=rk(S,D,R,AR,AN,AU,u,rem,p)
            g1=((e[0]+e[1]+e[2])-thr)*direction
            if g1<0:
                S,D,R,AR,AN,AU=e; t+=rem; rem=0.
                Npeak=max(Npeak,S+D+R)
                break
            if g0>=0:tau=0.;ee=(S,D,R,AR,AN,AU)
            else:
                lo=0.;hi=rem
                for k in range(60):
                    mid=.5*(lo+hi)
                    m=rk(S,D,R,AR,AN,AU,u,mid,p)
                    gm=((m[0]+m[1]+m[2])-thr)*direction
                    if gm>=0:hi=mid
                    else:lo=mid
                    if hi-lo<=1e-11:break
                tau=.5*(lo+hi)
                ee=rk(S,D,R,AR,AN,AU,u,tau,p)
            S,D,R,AR,AN,AU=ee;t+=tau;rem-=tau
            Npeak=max(Npeak,S+D+R);on=not on
            if on:
                n_activations+=1
                if np.isnan(first_on):first_on=t
            else:n_off+=1
            if tau<=1e-14 and rem>1e-14:
                eps=min(rem,1e-10)
                S,D,R,AR,AN,AU=rk(S,D,R,AR,AN,AU,1. if on else 0.,eps,p)
                t+=eps;rem-=eps;Npeak=max(Npeak,S+D+R)
        else:return np.full(11,np.nan)
    N=S+D+R
    return np.array([R/N if N>0 else np.nan,AR,AU,AN,N,Npeak,float(n_activations),first_on,t,float(n_off),1. if on else 0.])

def pack(row):
    r=row
    return np.array([r.rS,r.rD,r.rR,r.dS,r.dD,r.dR,r.alpha0,r.alpha1,r.beta,r.eta,1.,r.N_on,r.N_off],float)

def calc_prcc(df,idx):
    x=df.loc[idx,PARAMS+OUTS]
    z=np.column_stack([rankdata(x[c],method='average') for c in PARAMS]);z=np.column_stack([np.ones(len(idx)),z]);out=[]
    for k in OUTS:
        y=rankdata(x[k],method='average')
        for j,p in enumerate(PARAMS):
            others=np.delete(z,j+1,axis=1)
            xres=z[:,j+1]-others@np.linalg.lstsq(others,z[:,j+1],rcond=None)[0]
            yres=y-others@np.linalg.lstsq(others,y,rcond=None)[0]
            corr=float(np.corrcoef(xres,yres)[0,1]);out.append((k,p,corr,abs(corr)))
    return pd.DataFrame(out,columns=['output','parameter','PRCC','abs_PRCC'])

def main():
    samples=pd.read_csv(SAMPLE);node=pd.read_csv(ARCH)
    assert len(samples)==len(node)==2000
    assert np.allclose(samples[PARAMS].values,node[PARAMS].values,atol=1e-12,rtol=1e-12)
    initial_on=(samples.S0+samples.R0+samples.D0>=samples.N_on).astype(int)
    node["n_cycles_including_initial"]=node["n_cycles"].copy()
    node["n_cycles"]=node["n_cycles"]-initial_on
    node_idx=node.index[node.first_on.notna()]
    assert len(node_idx)==1628
    print('compile & cross-check baseline...',flush=True)
    # Compare event-localized integration implementations on 12 samples.
    controls=[0,1,2,3,4,5,6,7,8,9,10,11]
    cross=[]
    start=time.perf_counter()
    for i in controls:
        row=samples.iloc[i];p=pack(row)
        a=run(p,row.S0,0.,row.R0,row["T"],.1)
        b=grid_event_sim(p,row.S0,0.,row.R0,row["T"],.1)
        for ia,ib in [(0,0),(1,1),(2,3),(3,2),(4,4),(8,8)]:
            if not np.isclose(a[ia],b[ib],atol=2e-6,rtol=1e-7):raise RuntimeError(('EVENT CROSSCHECK FAIL',i,ia,ib,a[ia],b[ib]))
        cross.append(dict(row=i,phi_diff=a[0]-b[0],AR_diff=a[1]-b[1],Au_diff=a[2]-b[3],AN_diff=a[3]-b[2],t_diff=a[8]-b[8],Nmax=a[5], n_cycles=int(a[6])))
    pd.DataFrame(cross).to_csv(ROOT/'event_vs_grid_core_12.csv',index=False)
    print('crosscheck 12 passed; computing all 2000...',flush=True)
    arr=np.zeros((2000,11),float)
    for i,row in enumerate(samples.itertuples(index=False)):
        p=pack(row)
        arr[i]=run(p,row.S0,0.,row.R0,row.T,.1)
        if (i+1)%500==0:print('DONE',i+1,'seconds',round(time.perf_counter()-start,1),flush=True)
    out=samples.copy();out[OUTS]=arr[:,:8]
    out["n_cycles_including_initial"]=out["n_cycles"].copy()
    out["n_cycles"]=out["n_cycles"]-initial_on
    assert (out.n_cycles>=0).all()
    out['t_final']=arr[:,8];out['n_deactivations']=arr[:,9];out['final_on']=arr[:,10]
    out['failed']=(~np.isfinite(arr[:,:7]).all(axis=1)).astype(int)
    out.to_csv(ROOT/'lhs_results_event_2000.csv',index=False)
    assert int(out.failed.sum())==0
    assert np.max(np.abs(out.t_final-out["T"]))<1e-8
    event_idx=out.index[out.first_on.notna()]
    common_idx=node_idx.intersection(event_idx)
    pr_node_common=calc_prcc(node,node_idx)
    pr_event_node_subset=calc_prcc(out,node_idx) if out.loc[node_idx,'first_on'].notna().all() else None
    pr_event_active=calc_prcc(out,event_idx)
    pr_event_active.to_csv(ROOT/'prcc_event_15_predictors_activation_subset.csv',index=False)
    if pr_event_node_subset is not None:pr_event_node_subset.to_csv(ROOT/'prcc_event_on_common_activation_subset.csv',index=False)
    pr_node_common.to_csv(ROOT/'prcc_node_switched_activation_subset.csv',index=False)
    # Compare numerical switching protocols on a common activation subset.
    pr_node_same=calc_prcc(node,common_idx);pr_evt_same=calc_prcc(out,common_idx)
    merged=pr_node_same.merge(pr_evt_same,on=['output','parameter'],suffixes=('_nodal','_event'))
    merged['delta_PRCC']=merged.PRCC_event-merged.PRCC_nodal
    merged['delta_abs_PRCC']=merged.abs_PRCC_event-merged.abs_PRCC_nodal
    merged.to_csv(ROOT/'prcc_node_event_matched_subset.csv',index=False)
    paired=pd.DataFrame({'row_id':np.arange(len(node)),'node_activation':node.first_on.notna(),'event_activation':out.first_on.notna()})
    for col in OUTS:
        paired[col+'_nodal']=node[col];paired[col+'_event']=out[col];paired[col+'_delta']=out[col]-node[col]
    paired.to_csv(ROOT/'lhs_2000_cellwise_protocol_comparison.csv',index=False)
    top=pr_event_active.sort_values('abs_PRCC',ascending=False).groupby('output').head(5)
    top.to_csv(ROOT/'event_top5_by_output.csv',index=False)
    # Nmax step refinement on representative sample, separately from PRCC sampling
    fine=[]
    for i in [0,1,2,3,4,5,6,7,8,9,10,11,500,1000,1500,1999]:
        row=samples.iloc[i];p=pack(row)
        co=out.iloc[i];fi=run(p,row.S0,0.,row.R0,row["T"],.05)
        fine.append(dict(row_id=i,phi_diff=fi[0]-co.phi_R_T,AR_diff=fi[1]-co.A_R,Au_diff=fi[2]-co.A_u,AN_diff=fi[3]-co.A_N,Nmax_diff=fi[5]-co.N_max,cycles_diff=(fi[6]-int((row.S0+row.R0+row.D0)>=row.N_on))-co.n_cycles))
    pd.DataFrame(fine).to_csv(ROOT/'event_step_refinement_16.csv',index=False)
    metadata=dict(n=2000,n_initial_ON=int(initial_on.sum()),cycle_definition='postinitial OFF-to-ON activations for both switching protocols',first_on_definition='first treatment ON including t=0; differs from first reactivation diagnostic in 50x50 landscape',node_activation=int(node.first_on.notna().sum()),event_activation=int(out.first_on.notna().sum()),matched_activation=int(len(common_idx)),changed_activation=int((node.first_on.notna()!=out.first_on.notna()).sum()),n_failed=int(out.failed.sum()),max_T_error=float(np.max(np.abs(out.t_final-out["T"]))),max_matched_PRCC_abs_change=float(merged.delta_PRCC.abs().max()),max_cellwise_phi_change=float(np.max(abs(out.phi_R_T-node.phi_R_T))),max_cellwise_Au_change=float(np.max(abs(out.A_u-node.A_u))),elapsed_s=time.perf_counter()-start)
    (ROOT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('METADATA',json.dumps(metadata),flush=True)
    print('TOP OUTPUT',top.to_string(index=False),flush=True)
    print('LARGEST CHANGES',merged.sort_values('delta_PRCC',key=abs,ascending=False).head(15).to_string(index=False),flush=True)
if __name__=='__main__':main()
