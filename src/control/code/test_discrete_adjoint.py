import numpy as np,torch,csv
from pathlib import Path
from control_discrete_adjoint import exact_discrete_gradient,P0,IC,W,forward
import importlib.util
p=Path(__file__).resolve().parent/'discrete_gradient_check.py'
src=p.read_text();src=src[:src.index('rows=[]')];g={'__file__':str(p)};exec(src,g)
J= g['exact_gradient']
for scene,ws in [('A',0.),('B',0.),('C',0.),('B',1000.)]:
 w=W[scene].copy();w[7]=ws;w[8]=.52
 for init in ['AT','half']:
  rng=np.random.default_rng(102);u=np.full(24,.5) if init=='half' else rng.uniform(.0,1.,size=24)
  y,_=forward(u,.5,P0,w,IC);gn=exact_discrete_gradient(y,u,.5,P0,w);jc,gt=J(u,.5,P0,w)
  absmax=float(max(abs(gn-gt)));print('GRADIENT_TEST',scene,ws,init,absmax,abs(jc-(w[0]*y[-1,3]+w[1]*y[-1,4]+w[2]*y[-1,6]+w[7]*y[-1,7]+w[3]*sum(y[-1,:3])+w[4]*y[-1,2]+w[5]*y[-1,2]/(sum(y[-1,:3])+w[6]))),flush=True)
  assert absmax<1e-10
