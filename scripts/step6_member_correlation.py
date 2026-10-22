"""Spatial correlation of CFM member deviations before/after the untrained projections
(docs/step6_projection.md, W4/W5). Needs runs/step6/base.pt from run_step6_projection.py.

    python scripts/step6_member_correlation.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch.set_num_threads(1)
from pgcfm.data import PGCFMSequenceDataset, kai_data_config
from pgcfm.physics.primitive import ResidualScales
from pgcfm.projection import GradientFlowProjection, freeze
from pgcfm.train import take
from scripts.train_pgcfm import DEFAULTS, build
from scripts.run_step4_pgcfm import KEEP
from pgcfm import metrics
tr="runs/step6/data/train"; te="runs/step6/data/test"
mp,sp=f"{tr}/stats/global_mean.npy",f"{tr}/stats/global_std.npy"
d=kai_data_config(mp,sp)
ds=PGCFMSequenceDataset(d, te, horizon=1, train=False)
b=take(torch.utils.data.DataLoader(ds,batch_size=8),1)[0]
cfg=dict(DEFAULTS, model_channels=32, channel_mult=[1,2,2], num_res_blocks=1)
m=build(cfg, ds.img_shape_x, ds.img_shape_y, np.load(mp), np.load(sp))
ck=torch.load("runs/step6/base.pt"); m.load_state_dict(ck["model"]); m.guide.scales=ResidualScales(**ck["scales"]); freeze(m.guide)
M=8; x,y,st,dn=b["x0"],b["targets"][:,0],b["static"],b["doy"][:,1]
rep=lambda t:t.repeat(M,*([1]*(t.dim()-1)))
g=torch.Generator().manual_seed(3)
with torch.no_grad(): raw=m.cfm.sample(rep(x),rep(st),rep(dn),steps=8,generator=g)
def lag1(e):  # e (M,B,C,H,W): lon-lag-1 correlation of member deviations; and of truth tendency
    dev=e-e.mean(0,keepdim=True)
    a,bb=dev, torch.roll(dev,1,-1)
    return float((a*bb).mean()/ (a*a).mean())
def vlag(e):  # vertical neighbour correlation of member deviations inside z,t,u blocks (12 levels)
    dev=(e-e.mean(0,keepdim=True))[:,:,:60].reshape(*e.shape[:2],5,12,*e.shape[-2:])
    a,bb=dev[:,:,:,1:],dev[:,:,:,:-1]
    return float((a*bb).mean()/((a*a).mean()*(bb*bb).mean()).sqrt())
out={}
tend=(y-x)[None]
out["truth_tendency_lon_lag1"]=float((tend*torch.roll(tend,1,-1)).mean()/(tend*tend).mean())
for name,f in [("cfm_raw",None),("gradflow",GradientFlowProjection(m.cfm,m.guide)),("gradflow_smooth",GradientFlowProjection(m.cfm,m.guide,control="smooth"))]:
    e=raw if f is None else f.project(rep(x),raw,rep(st),rep(dn))
    e=e.view(M,*x.shape)
    out[name]={"lon_lag1":lag1(e),"vert_lag1":vlag(e),"spread":float(metrics.ens_spread(e[:,:,KEEP]).pow(2).mean().sqrt())}
print(json.dumps(out,indent=1))
with open("runs/step6/member_correlation.json", "w") as f:
    json.dump(out, f, indent=1)
