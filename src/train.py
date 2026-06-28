#!/usr/bin/env python3
"""
Review Residuals --- training and evaluation for the paper
"An Update-Conditioned Residual Gate Whose Advantage Emerges at Scale" (Kramer, 2026).

Trains, from scratch on TinyStories, three identity-preserving (additive) residual variants:
  - review_neutral : update scaled by a gate conditioned on BOTH state and proposed update  (ours)
  - highway        : update scaled by a gate conditioned on the state only  (param-matched up)
  - standard       : plain residual, update added with coefficient 1        (param-matched up)
across five model sizes (60M-1B). Resumable; writes per-run validation losses to scaling_v8.csv.
Requires a CUDA GPU (80GB for the 590M/1B sizes). See README.md to reproduce.
"""

#!/usr/bin/env python3
# Review Residuals scaling sweep — RESILIENT, RESUMABLE, disconnect-proof.
# Run on the pod:   nohup python run_scaling.py > scaling.log 2>&1 &
# Watch progress:   tail -f scaling.log
# It resumes automatically: any run already in scaling_partial.csv is skipped.
# Each run is launched in its OWN subprocess, so even a hard crash on one model
# only loses that one model — the sweep keeps going.

import os, sys, csv, time, math, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
CSV="scaling_v8.csv"

# ---- the scaling ladder (identical to the notebook) ----
SIZES=[
  dict(name="1B",   d=1536, L=24, h=16, steps=6000, seeds=[0,1,2]),
  dict(name="590M", d=1280, L=20, h=20, steps=7000, seeds=[0,1,2]),
  dict(name="320M", d=1024, L=16, h=16, steps=8000, seeds=[0,1]),
  dict(name="150M", d=768,  L=12, h=12, steps=8000, seeds=[0,1,2]),
  dict(name="60M",  d=512,  L=8,  h=8,  steps=8000, seeds=[0,1,2]),
]
VARIANTS=["review_neutral","highway","standard"]
# AttnRes is ~8x slower than the others; only run it at the small sizes as anchors.
ATTNRES_SIZES=[]   # no attnres in the recipe-fixed sweep
BLOCK=256; BATCH=64; LR=2e-4; WARMUP=500; N_TEXT=400000   # lower LR + warmup (fixes large-scale divergence)
FIELDS=["size","variant","seed","params_M","steps","val_loss","ece","minutes"]

def done_set():
    s=set()
    if os.path.exists(CSV):
        with open(CSV) as f:
            for r in csv.DictReader(f):
                s.add((r["size"],r["variant"],int(r["seed"])))
    return s

def append_row(row):
    new = not os.path.exists(CSV)
    with open(CSV,"a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(row)

# ============================================================ ORCHESTRATOR
def orchestrate():
    print("[orch] installing deps...",flush=True)
    subprocess.run([sys.executable,"-m","pip","install","-q","datasets","transformers","accelerate","matplotlib","pandas"])
    done=done_set()
    plan=[]
    for SZ in SIZES:
        for v in VARIANTS:
            if v=="attnres_plus":
                if SZ["name"] not in ATTNRES_SIZES: continue
                seeds = SZ["seeds"] if SZ["name"]=="60M" else [0]
            else:
                seeds = SZ["seeds"]
            for sd in seeds: plan.append((SZ,v,sd))
    todo=[(SZ,v,sd) for (SZ,v,sd) in plan if (SZ["name"],v,sd) not in done]
    print(f"[orch] {len(done)} runs already done, {len(todo)} to go",flush=True)
    t0=time.time()
    for SZ,v,sd in todo:
        tag=f"{SZ['name']}/{v}/seed{sd}"
        print(f"\n[orch] === launching {tag} (elapsed {(time.time()-t0)/3600:.2f}h) ===",flush=True)
        # isolate each run in a fresh process: a hard crash here cannot kill the sweep
        rc=subprocess.call([sys.executable, os.path.abspath(__file__), "--worker", SZ["name"], v, str(sd)])
        if rc!=0:
            print(f"[orch] !! {tag} exited with code {rc} (logged as failure, continuing)",flush=True)
        else:
            print(f"[orch] ok {tag}",flush=True)
    print(f"\n[orch] SWEEP COMPLETE in {(time.time()-t0)/3600:.2f}h. Results in {CSV}.",flush=True)
    try:
        make_plot()
    except Exception as e:
        print("[orch] plot skipped:",e,flush=True)

# ============================================================ WORKER (one run)
def worker(size_name,variant,seed):
    import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
    from datasets import load_dataset; from transformers import GPT2TokenizerFast
    torch.set_float32_matmul_precision("high"); torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    device="cuda" if torch.cuda.is_available() else "cpu"; assert device=="cuda","need GPU"
    SZ=[s for s in SIZES if s["name"]==size_name][0]
    tok=GPT2TokenizerFast.from_pretrained("gpt2"); VOCAB=tok.vocab_size

    class RMSNorm(nn.Module):
        def __init__(s,d): super().__init__(); s.g=nn.Parameter(torch.ones(d))
        def forward(s,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-5)*s.g
    class Attn(nn.Module):
        def __init__(s,d,h,block): super().__init__(); s.h=h; s.qkv=nn.Linear(d,3*d); s.proj=nn.Linear(d,d)
        def forward(s,x):
            B,T,d=x.shape; q,k,v=s.qkv(x).split(d,2)
            q=q.view(B,T,s.h,d//s.h).transpose(1,2); k=k.view(B,T,s.h,d//s.h).transpose(1,2); v=v.view(B,T,s.h,d//s.h).transpose(1,2)
            return s.proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(B,T,d))
    class MLP(nn.Module):
        def __init__(s,d): super().__init__(); s.f1=nn.Linear(d,4*d); s.f2=nn.Linear(4*d,d)
        def forward(s,x): return s.f2(F.gelu(s.f1(x)))
    def is_attnres(v): return v in ("attnres","attnres_plus")
    class GPT(nn.Module):
        def __init__(s,variant,d,n_layer,n_head,block,vocab):
            super().__init__(); s.variant=variant
            s.tok=nn.Embedding(vocab,d); s.pos=nn.Embedding(block,d); s.norms=nn.ModuleList(); s.subs=nn.ModuleList()
            for i in range(2*n_layer):
                s.norms.append(RMSNorm(d)); s.subs.append(Attn(d,n_head,block) if i%2==0 else MLP(d))
            nS=2*n_layer
            if variant=="highway": s.gate=nn.ModuleList([nn.Linear(d,d) for _ in range(nS)])
            if variant=="review_neutral":
                s.rgate=nn.ModuleList([nn.Linear(2*d,d) for _ in range(nS)])
                for g in s.rgate: nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)
            if variant=="layerscale": s.ls=nn.ParameterList([nn.Parameter(torch.ones(d)*0.1) for _ in range(nS)])
            if variant=="rezero":     s.rez=nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(nS)])
            if is_attnres(variant): s.dq=nn.Parameter(torch.randn(nS+1,d)*0.02); s.dk=nn.Linear(d,d,bias=False)
            s.lnf=RMSNorm(d); s.head=nn.Linear(d,vocab,bias=False); s.head.weight=s.tok.weight
            # --- GPT-2 / nanoGPT initialization (stable deep training) ---
            def _gpt2(mod):
                if isinstance(mod,nn.Linear):
                    nn.init.normal_(mod.weight,mean=0.0,std=0.02)
                    if mod.bias is not None: nn.init.zeros_(mod.bias)
                elif isinstance(mod,nn.Embedding):
                    nn.init.normal_(mod.weight,mean=0.0,std=0.02)
            s.apply(_gpt2)
            # scale residual-projection outputs by 1/sqrt(2*n_layer)  <-- the key deep-stability fix
            for _n,_p in s.named_parameters():
                if _n.endswith("proj.weight") or _n.endswith("f2.weight"):
                    nn.init.normal_(_p,mean=0.0,std=0.02/math.sqrt(2*n_layer))
            # keep the review gate neutral (must stay zero for r=0.5 start)
            if variant=="review_neutral":
                for g in s.rgate: nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)
        def _rms(s,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-5)
        def _depth_attn(s,M,qi):
            K=s._rms(s.dk(M)); a=(K*qi.view(1,1,1,-1)).sum(-1).softmax(-1).unsqueeze(-1); return (a*M).sum(2)
        def forward(s,idx,targets=None):
            B,T=idx.shape; x0=s.tok(idx)+s.pos(torch.arange(T,device=idx.device))[None]
            if is_attnres(s.variant):
                mem=[x0]
                for i,(nrm,sub) in enumerate(zip(s.norms,s.subs)):
                    mem.append(sub(nrm(s._depth_attn(torch.stack(mem,2),s.dq[i]))))
                h=s._depth_attn(torch.stack(mem,2),s.dq[-1])
            else:
                h=x0
                for i,(nrm,sub) in enumerate(zip(s.norms,s.subs)):
                    u=sub(nrm(h))
                    if s.variant=="highway": g=torch.sigmoid(s.gate[i](h)); h=h+g*u            # additive (identity preserved)
                    elif s.variant=="review_neutral": r=torch.sigmoid(s.rgate[i](torch.cat([s._rms(h),s._rms(u)],-1))); h=h+r*u   # additive (identity preserved)
                    elif s.variant=="layerscale": h=h+s.ls[i]*u
                    elif s.variant=="rezero": h=h+s.rez[i]*u
                    else: h=h+u
            logits=s.head(s.lnf(h))
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) if targets is not None else None
            return logits,loss

    def attnres_width(SZ):
        def est(var,d,L,h):
            nS=2*L; p=VOCAB*d+BLOCK*d+nS*d+L*(4*d*d+4*d)+L*(8*d*d+5*d)+d
            if var=="review_neutral": p+=nS*(2*d*d+d)
            elif var=="attnres_plus": p+=(nS+1)*d+d*d
            return p
        base=est("review_neutral",SZ["d"],SZ["L"],SZ["h"]); d=SZ["d"]
        while est("attnres_plus",d,SZ["L"],SZ["h"])<base: d+=SZ["h"]
        return d
    def highway_width(SZ):
        def est(var,d,L,h):
            nS=2*L; p=VOCAB*d+BLOCK*d+nS*d+L*(4*d*d+4*d)+L*(8*d*d+5*d)+d
            if var=="review_neutral": p+=nS*(2*d*d+d)
            elif var=="highway": p+=nS*(d*d+d)
            return p
        base=est("review_neutral",SZ["d"],SZ["L"],SZ["h"]); d=SZ["d"]
        while est("highway",d,SZ["L"],SZ["h"])<base: d+=SZ["h"]
        return d
    def standard_width(SZ):
        def est(var,d,L,h):
            nS=2*L; p=VOCAB*d+BLOCK*d+nS*d+L*(4*d*d+4*d)+L*(8*d*d+5*d)+d
            if var=="review_neutral": p+=nS*(2*d*d+d)
            return p
        base=est("review_neutral",SZ["d"],SZ["L"],SZ["h"]); d=SZ["d"]
        while est("standard",d,SZ["L"],SZ["h"])<base: d+=SZ["h"]
        return d

    def load_data():
        texts=load_dataset("roneneldan/TinyStories", split=f"train[:{N_TEXT}]")["text"]
        ids=[]
        for i in range(0,len(texts),2000):
            for e in tok(texts[i:i+2000])["input_ids"]: ids.extend(e); ids.append(tok.eos_token_id)
        data=np.array(ids,dtype=np.uint16); sp=int(len(data)*0.97)
        return torch.from_numpy(data[:sp].astype(np.int64)), torch.from_numpy(data[sp:].astype(np.int64))
    def get_batch(t,B,T):
        ix=np.random.randint(0,len(t)-T-1,size=B)
        x=torch.stack([t[i:i+T] for i in ix]); y=torch.stack([t[i+1:i+1+T] for i in ix])
        return x.to(device,non_blocking=True), y.to(device,non_blocking=True)
    @torch.no_grad()
    def evaluate(model,val_t,B,n=80):
        model.eval(); L=[]; C=[]; K=[]
        for _ in range(n):
            x,y=get_batch(val_t,B,BLOCK); lo,l=model(x,y); L.append(l.item())
            p=lo.softmax(-1); c,pr=p.max(-1); C.append(c.flatten().cpu().numpy()); K.append((pr==y).flatten().cpu().numpy())
        C=np.concatenate(C); K=np.concatenate(K).astype(float); e=np.linspace(0,1,16); ece=0
        for i in range(15):
            m=(C>e[i])&(C<=e[i+1])
            if m.sum(): ece+=m.sum()/len(C)*abs(K[m].mean()-C[m].mean())
        return float(np.mean(L)),float(ece)

    print(f"[worker] {size_name} {variant} seed{seed} on {torch.cuda.get_device_name(0)}",flush=True)
    train_t,val_t=load_data()
    d = highway_width(SZ) if variant=="highway" else (standard_width(SZ) if variant=="standard" else SZ["d"])  # param-match baselines UP
    batch=BATCH
    for attempt in range(4):
        try:
            torch.manual_seed(seed); np.random.seed(seed)
            m=GPT(variant,d,SZ["L"],SZ["h"],BLOCK,VOCAB).to(device)
            P=sum(p.numel() for p in m.parameters())/1e6
            opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0.1,betas=(0.9,0.95))
            def _lrlam(step, T=SZ["steps"], W=WARMUP):
                if step < W: return (step+1)/W                       # linear warmup
                prog=(step-W)/max(1,(T-W)); return 0.1+0.9*0.5*(1+math.cos(math.pi*min(1.0,prog)))  # cosine to 10%
            sch=torch.optim.lr_scheduler.LambdaLR(opt,_lrlam); t0=time.time()
            for step in range(SZ["steps"]):
                x,y=get_batch(train_t,batch,BLOCK); _,loss=m(x,y)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); sch.step()
                if step%1000==0: print(f"    step {step}/{SZ['steps']} loss {loss.item():.3f}",flush=True)
            vl,ece=evaluate(m,val_t,batch); mins=(time.time()-t0)/60
            append_row(dict(size=size_name,variant=variant,seed=seed,params_M=round(P,2),
                            steps=SZ["steps"],val_loss=round(vl,4),ece=round(ece,4),minutes=round(mins,1)))
            print(f"[worker] DONE {size_name} {variant} seed{seed} {P:.1f}M val {vl:.4f} ece {ece:.4f} {mins:.1f}min",flush=True)
            return 0
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower() and batch>8:
                torch.cuda.empty_cache(); batch//=2
                print(f"[worker] OOM -> retry at batch {batch}",flush=True)
            else:
                raise
    return 1

# ============================================================ PLOT
def make_plot():
    import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    df=pd.read_csv(CSV)
    df=df[df["val_loss"]<4.5]   # drop diverged runs from the plot
    agg=df.groupby(["size","variant"]).agg(params_M=("params_M","mean"),val_loss=("val_loss","mean")).reset_index()
    order=[s["name"] for s in SIZES]; agg["o"]=agg["size"].map({n:i for i,n in enumerate(order)}); agg=agg.sort_values("o")
    col={'review_neutral':'#27ae60','attnres_plus':'#8e44ad','highway':'#999'}
    mk={'review_neutral':'o','attnres_plus':'s','highway':'^'}
    fig,ax=plt.subplots(figsize=(8,5.4))
    for v in VARIANTS:
        s=agg[agg.variant==v].sort_values("params_M")
        if len(s): ax.plot(s["params_M"],s["val_loss"],marker=mk[v],color=col[v],lw=2,ms=8,label=v)
    ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)")
    ax.set_ylabel("validation loss  (lower = better)")
    ax.set_title("Review Residuals scaling — loss vs parameters (TinyStories)")
    ax.grid(alpha=.3,which="both"); ax.legend()
    plt.tight_layout(); plt.savefig("scaling_result.png",dpi=140)
    print("[plot] wrote scaling_result.png",flush=True)

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--worker":
        sys.exit(worker(sys.argv[2], sys.argv[3], int(sys.argv[4])))
    else:
        orchestrate()
