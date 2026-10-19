from __future__ import annotations

import math

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


def _bit(idx: torch.Tensor, w: int, n: int) -> torch.Tensor:
    return (idx >> (n - 1 - w)) & 1


def cnot_perm(pairs, n: int) -> torch.Tensor:
    N = 1 << n
    idx = torch.arange(N)
    perm = idx.clone()
    for c, t in pairs:
        f = torch.where(_bit(idx, c, n) == 1, idx ^ (1 << (n - 1 - t)), idx)
        perm = perm[f]
    return perm


def z_signs(n: int) -> torch.Tensor:
    idx = torch.arange(1 << n)
    return torch.stack([1.0 - 2.0 * _bit(idx, w, n).to(torch.get_default_dtype()) for w in range(n)], dim=1)


def product_state(theta: torch.Tensor, plus: bool) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    if plus:
        v0, v1 = (c - s) / math.sqrt(2), (s + c) / math.sqrt(2)
    else:
        v0, v1 = c, s
    v = torch.stack([v0, v1], dim=-1)  # (..., n, 2)
    psi = v[..., 0, :]
    for w in range(1, theta.shape[-1]):
        psi = (psi.unsqueeze(-1) * v[..., w, None, :]).reshape(*psi.shape[:-1], -1)
    return psi


def apply_ry(psi: torch.Tensor, theta: torch.Tensor, w: int, n: int) -> torch.Tensor:
    L, R = 1 << w, 1 << (n - 1 - w)
    s = psi.reshape(*psi.shape[:-1], L, 2, R)
    a0, a1 = s[..., 0, :], s[..., 1, :]
    c = torch.cos(theta / 2)[..., None, None]
    si = torch.sin(theta / 2)[..., None, None]
    out = torch.stack([c * a0 - si * a1, si * a0 + c * a1], dim=-2)
    return out.reshape(*out.shape[:-3], -1)


def rz_layer_phase(phi: torch.Tensor, n: int, signs: torch.Tensor) -> torch.Tensor:
    ang = -0.5 * (signs @ phi)  # (N,)
    return torch.polar(torch.ones_like(ang), ang)


def expval_z(psi: torch.Tensor, signs: torch.Tensor, wires) -> torch.Tensor:
    p = psi.real ** 2 + psi.imag ** 2 if psi.is_complex() else psi ** 2
    return p @ signs[:, list(wires)]


class QLSTM(nn.Module):

    def __init__(self, n_in=4, H=3, depth=1, n_out=1):
        super().__init__()
        self.H, self.depth = H, depth
        nq = self.nq = n_in + H
        self.w = nn.ParameterList([nn.Parameter(torch.empty(depth, nq).uniform_(0, 2 * math.pi)) for _ in range(4)])
        self.out = nn.Linear(H, n_out)
        ladder = [(w, w + 1) for w in range(0, nq - 1, 2)] + [(w, w + 1) for w in range(1, nq - 1, 2)]
        self.register_buffer("perm", cnot_perm(ladder, nq), persistent=False)
        self.register_buffer("signs", z_signs(nq), persistent=False)

    def gates(self, v):
        W = torch.stack(list(self.w), dim=0)  # (4, depth, nq)
        psi = product_state(v, plus=True).unsqueeze(0)  # (1, B, N)
        for d in range(self.depth):
            psi = psi[..., self.perm]
            for q in range(self.nq):
                psi = apply_ry(psi, W[:, d, q][:, None], q, self.nq)  # (4, B, N)
        return expval_z(psi, self.signs, range(self.H))

    def forward(self, X):
        B, L, _ = X.shape
        h = X.new_zeros(B, self.H)
        c = X.new_zeros(B, self.H)
        for t in range(L):
            g = self.gates(torch.cat([X[:, t, :], h], dim=-1))
            f, i, gg, o = torch.sigmoid(g[0]), torch.sigmoid(g[1]), torch.tanh(g[2]), torch.sigmoid(g[3])
            c = f * c + i * gg
            h = o * torch.tanh(c)
        return self.out(h)


class QFWP(nn.Module):

    def __init__(self, s_dim=4, latent=3, nq=3, D=1, A=1):
        super().__init__()
        self.nq, self.D, self.A = nq, D, A
        self.enc = nn.Linear(s_dim, latent)
        self.lay = nn.Linear(latent, D)
        self.qub = nn.Linear(latent, nq)
        self.post = nn.Linear(A, 1)
        self.register_buffer("perm", cnot_perm([(q, q + 1) for q in range(nq - 1)], nq), persistent=False)
        self.register_buffer("signs", z_signs(nq), persistent=False)

    def circ(self, z, theta):
        psi = product_state(z, plus=False)
        for d in range(self.D):
            psi = psi[..., self.perm]
            for q in range(self.nq):
                psi = apply_ry(psi, theta[:, d * self.nq + q], q, self.nq)
        return expval_z(psi, self.signs, range(self.A))

    def forward(self, X):
        B, L, _ = X.shape
        theta = X.new_zeros(B, self.D * self.nq)
        y = None
        for t in range(L):
            z = self.enc(X[:, t, :])
            a = self.lay(z).unsqueeze(2)
            b = self.qub(z).unsqueeze(1)
            theta = theta + (a * b).reshape(B, self.D * self.nq)
            y = self.post(self.circ(z, theta))
        return y


class FWPClassical(QFWP):

    def circ(self, z, theta):
        return torch.tanh((z * theta[:, : self.nq]).sum(-1, keepdim=True))


class LSTM(nn.Module):

    def __init__(self, n_in=4, H=3, n_out=1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, H, batch_first=True)
        self.out = nn.Linear(H, n_out)

    def forward(self, X):
        o, _ = self.lstm(X)
        return self.out(o[:, -1, :])


def _init_normal(*layers):
    for l in layers:
        nn.init.normal_(l.weight, mean=0.0, std=0.1)
        nn.init.constant_(l.bias, 0.0)


class _VQCWeights(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(*shape).uniform_(0, 2 * math.pi))


class QNet(nn.Module):

    def __init__(self, s_dim=10, a_dim=3, nq=8):
        super().__init__()
        self.nq = nq
        self.pa, self.pc = nn.Linear(s_dim, nq), nn.Linear(s_dim, nq)
        self.va, self.vc = _VQCWeights((2, nq)), _VQCWeights((2, nq))
        self.ha, self.hc = nn.Linear(nq, a_dim), nn.Linear(nq, 1)
        _init_normal(self.pa, self.pc, self.ha, self.hc)
        self.register_buffer("perm", cnot_perm([(q, (q + 1) % nq) for q in range(nq)], nq), persistent=False)
        self.register_buffer("signs", z_signs(nq), persistent=False)

    def vqc(self, x, w):
        psi = product_state(x, plus=True).to(torch.complex128)
        psi = psi[..., self.perm] * rz_layer_phase(w[0], self.nq, self.signs)
        psi = psi[..., self.perm]
        for q in range(self.nq):
            psi = apply_ry(psi, w[1, q], q, self.nq)
        return expval_z(psi, self.signs, range(self.nq))

    def refresh(self):
        with torch.no_grad():
            w = self.va.weights
            M = torch.eye(1 << self.nq, dtype=torch.complex128, device=w.device)
            M = M[..., self.perm] * rz_layer_phase(w[0], self.nq, self.signs)
            M = M[..., self.perm]
            for q in range(self.nq):
                M = apply_ry(M, w[1, q], q, self.nq)
            self._M = M

    def pi_fast(self, s):
        with torch.no_grad():
            psi = product_state(torch.tanh(self.pa(s)), plus=True).to(torch.complex128) @ self._M
            return self.ha(expval_z(psi, self.signs, range(self.nq)))

    def pi(self, s):
        return self.ha(self.vqc(torch.tanh(self.pa(s)), self.va.weights))

    def v(self, s):
        return self.hc(self.vqc(torch.tanh(self.pc(s)), self.vc.weights))


class CNet(nn.Module):

    def __init__(self, s_dim=10, a_dim=3, h=128):
        super().__init__()
        self.pi1, self.pi2 = nn.Linear(s_dim, h), nn.Linear(h, a_dim)
        self.v1, self.v2 = nn.Linear(s_dim, h), nn.Linear(h, 1)
        _init_normal(self.pi1, self.pi2, self.v1, self.v2)

    def pi(self, s):
        return self.pi2(torch.tanh(self.pi1(s)))

    def v(self, s):
        return self.v2(torch.tanh(self.v1(s)))


MODELS = {"QLSTM": QLSTM, "QFWP": QFWP, "FWPC": FWPClassical, "LSTM": LSTM}
AGENTS = {"QA3C": QNet, "A3C": CNet, "A3C_small": lambda: CNet(h=9)}


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))
