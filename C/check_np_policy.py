import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "qfin-main"))
sys.path.insert(0, str(HERE.parent / "B"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import cdata
import fastsim as F
import pl_models as P
from env_c import EnvC
from train_c import load_probs, np_policy, sample_masked


class FixedU:

    def __init__(self, u):
        self.u = u

    def random(self):
        return self.u


def masked_probs(lg, m):
    lg = np.where(m, lg, -np.inf)
    p = np.exp(lg - lg.max())
    return p / p.sum()


torch.set_num_threads(1)
df = cdata.load_data()
tr, _, _ = cdata.split(df, 2020)
probs = load_probs("qlstm", 2020, df)
out = []
for seed in (0, 1, 2):
    torch.manual_seed(seed); pl = P.QNet()
    torch.manual_seed(seed); fs = F.QNet()
    g = torch.Generator().manual_seed(100 + seed)
    with torch.no_grad():
        for p in pl.parameters():
            p.add_(0.5 * torch.randn(p.shape, generator=g))
    fs.load_state_dict(pl.state_dict())
    d_param = max((a - b).abs().max().item() for a, b in zip(pl.parameters(), fs.parameters()))
    pol = np_policy(fs)
    env = EnvC(df[tr], probs[tr], 0.0005, "logret")
    rng = np.random.default_rng(seed)
    s, done = env.reset(), False
    n = mism = 0
    d_logit = d_prob = 0.0
    acts = np.zeros(3, int)
    while not done:
        m = env.mask()
        u = rng.random()
        with torch.no_grad():
            lg_pl = pl.pi(torch.from_numpy(s))[0].numpy()
        lg_np = pol(s[0])
        d_logit = max(d_logit, float(np.abs(lg_pl - lg_np).max()))
        d_prob = max(d_prob, float(np.abs(masked_probs(lg_pl, m) - masked_probs(lg_np, m)).max()))
        a_pl, a_np = sample_masked(lg_pl, m, FixedU(u)), sample_masked(lg_np, m, FixedU(u))
        mism += int(a_pl != a_np)
        acts[a_pl] += 1
        n += 1
        _, done = env.step(a_pl)
        if not done:
            s = env.state()
    r = {"seed": seed, "steps": n, "action_mismatches": mism, "max_param_diff": d_param, "max_logit_diff": d_logit,
         "max_prob_diff": d_prob, "action_counts": acts.tolist(),
         "final_return_pct": float(100 * (env.values[-1] / env.init - 1))}
    out.append(r)
    print(f"種子 {seed}：{n} 步，PennyLane 與 NumPy 選的動作不同 {mism} 次；logits 最大差 {d_logit:.1e}、"
          f"動作機率最大差 {d_prob:.1e}；動作次數（持有／買／賣）{acts.tolist()}", flush=True)
(HERE / "results" / "check_np_policy.json").write_text(json.dumps(out, indent=1))
