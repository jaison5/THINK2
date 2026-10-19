from __future__ import annotations

import math

import numpy as np


class EnvC:
    def __init__(self, df, probs, cost=0.0005, reward="logret", units0=10):
        self.p = df["Close"].to_numpy()
        self.ma5, self.ma20, self.ma60 = (df[f"MA{n}"].to_numpy() for n in (5, 20, 60))
        self.volrel = df["VOLREL"].to_numpy()
        self.pb, self.pr = probs[:, 1], probs[:, 0]
        self.c, self.reward, self.units0 = cost, reward, units0
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = self.units0 * self.p[0]
        self.init = self.cash
        self.units = 0
        self.avg = 0.0
        self.trades, self.buys, self.cost_paid = [], 0, 0.0
        self.values = [self.cash]
        return self.state()

    def value(self, p):
        return self.cash + self.units * p

    def state(self):
        t, p = self.t, self.p[self.t]
        v = self.value(p)
        hold = self.units > 0
        upnl = (p / self.avg - 1) * 100 if hold else 0.0
        apr = p / self.avg if hold else 1.0
        return np.array([[self.pb[t], self.pr[t], self.cash / v, self.units * p / v, upnl, apr,
                          (p / self.ma20[t] - 1) * 100, (p / self.ma60[t] - 1) * 100,
                          (self.ma5[t] / self.ma20[t] - 1) * 100, self.volrel[t]]])

    def mask(self):
        p = self.p[self.t]
        return np.array([True, self.cash >= p * (1 + self.c), self.units > 0])

    def step(self, a):
        t, p, c = self.t, self.p[self.t], self.c
        v0 = self.value(p)
        paper = self.reward == "paper"
        r = -0.02 if paper else 0.0
        if a == 1 and self.cash >= p * (1 + c):
            pay = p * (1 + c)
            if paper:
                r += 0.5 if p > self.ma20[t] else -2.0
            self.avg = (self.avg * self.units + pay) / (self.units + 1)
            self.units += 1
            self.cash -= pay
            self.cost_paid += p * c
            self.buys += 1
        elif a == 2 and self.units > 0:
            pnl = (p * (1 - c) / self.avg - 1) * 100
            self.trades.append(pnl)
            if paper:
                r += (10 + 50 * pnl) if pnl > 0 else (-2 - 10 * abs(pnl))
            self.cash += self.units * p * (1 - c)
            self.cost_paid += self.units * p * c
            self.units, self.avg = 0, 0.0
        self.t += 1
        p2 = self.p[self.t]
        v1 = self.value(p2)
        if paper:
            if self.units > 0:
                upnl = (p2 / self.avg - 1) * 100
                if upnl < 0:
                    r -= 5 * upnl ** 2
            r = min(max(r, -15.0), 30.0)
        else:
            r = 100 * math.log(v1 / v0)
        self.values.append(v1)
        return r, self.t >= len(self.p) - 1


def evaluate(policy, df, probs, cost):
    env = EnvC(df, probs, cost, "logret")
    s, done, acts, held = env.reset(), False, [], []
    while not done:
        a = int(policy(s, env.mask()))
        acts.append(a)
        _, done = env.step(a)
        held.append(env.units > 0)
        if not done:
            s = env.state()
    v = np.array(env.values)
    tr = np.array(env.trades)
    dr = np.diff(v) / v[:-1]
    return {"start": str(df.index[0].date()), "end": str(df.index[-1].date()), "days": int(len(df)), "cost": cost,
            "total_return_pct": float(100 * (v[-1] / v[0] - 1)),
            "max_drawdown_pct": float(100 * np.max(1 - v / np.maximum.accumulate(v))),
            "sharpe_ann": float(np.sqrt(252) * dr.mean() / dr.std()) if dr.std() > 0 else 0.0,
            "trades": int(len(tr)), "buys": int(env.buys),
            "win_rate_pct": float(100 * np.mean(tr > 0)) if len(tr) else None,
            "avg_trade_pct": float(tr.mean()) if len(tr) else None,
            "time_in_market_pct": float(100 * np.mean(held)), "cost_paid_pct": float(100 * env.cost_paid / v[0]),
            "action_counts": np.bincount(acts, minlength=3).tolist(),
            "buy_hold_pct": float(100 * (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1))}


def policy_buy_hold(s, m):
    return 1 if m[1] else 0


def make_policy_ma(df):
    up = (df["MA5"] > df["MA20"]).to_numpy()
    state = {"t": 0}

    def pol(s, m):
        t = state["t"]; state["t"] += 1
        if up[t]:
            return 1 if m[1] else 0
        return 2 if m[2] else 0
    return pol


def make_policy_random(seed):
    rng = np.random.default_rng(seed)

    def pol(s, m):
        return int(rng.choice(np.flatnonzero(m)))
    return pol
